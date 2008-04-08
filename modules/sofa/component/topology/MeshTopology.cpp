/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <iostream>
#include <sofa/helper/io/Mesh.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/helper/io/MeshTopologyLoader.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/fixed_array.h>
#include <set>
#include <string.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(MeshTopology)

int MeshTopologyClass = core::RegisterObject("Generic mesh topology")
        .addAlias("Mesh")
        .add< MeshTopology >()
        ;

MeshTopology::MeshTopology()
    : nbPoints(0)
    , seqEdges(initData(&seqEdges,"lines","List of line indices")), validEdges(false)
    , seqTriangles(initData(&seqTriangles,"triangles","List of triangle indices")), validTriangles(false)
    , validQuads(false), validTetras(false), validHexas(false), revision(0)
    , filename(initData(&filename,"filename","Filename of the object"))
{
}


class MeshTopology::Loader : public helper::io::MeshTopologyLoader
{
public:
    MeshTopology* dest;
    Loader(MeshTopology* dest) : dest(dest) {}
    virtual void addPoint(double px, double py, double pz)
    {
        dest->seqPoints.push_back(helper::make_array(px, py, pz));
        if (dest->seqPoints.size() > (unsigned)dest->nbPoints)
            dest->nbPoints = dest->seqPoints.size();
    }
    virtual void addEdge(int p1, int p2)
    {
        dest->seqEdges.beginEdit()->push_back(Edge(p1,p2));
        dest->seqEdges.endEdit();
    }
    virtual void addTriangle(int p1, int p2, int p3)
    {
        dest->seqTriangles.beginEdit()->push_back(Triangle(p1,p2,p3));
        dest->seqTriangles.endEdit();
    }
    virtual void addQuad(int p1, int p2, int p3, int p4)
    {
        dest->seqQuads.push_back(Quad(p1,p2,p3,p4));
    }
    virtual void addTetra(int p1, int p2, int p3, int p4)
    {
        dest->seqTetras.push_back(Tetra(p1,p2,p3,p4));
    }
    virtual void addCube(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8)
    {
        //dest->seqHexas.push_back(Hexa(p1,p2,p3,p4,p5,p6,p7,p8));
        dest->seqHexas.push_back(Hexa(p1,p2,p4,p3,p5,p6,p8,p7));
    }
};

void MeshTopology::clear()
{
    nbPoints = 0;
    seqEdges.beginEdit()->clear(); seqEdges.endEdit();
    seqTriangles.beginEdit()->clear(); seqTriangles.endEdit();
    seqQuads.clear();
    seqTetras.clear();
    seqHexas.clear();
    invalidate();
}

bool MeshTopology::load(const char* filename)
{
    clear();
    Loader loader(this);

    if ((strlen(filename)>4 && !strcmp(filename+strlen(filename)-4,".obj"))
        || (strlen(filename)>6 && !strcmp(filename+strlen(filename)-6,".trian")))
    {
        helper::io::Mesh* mesh = helper::io::Mesh::Create(filename);
        if (mesh==NULL) return false;

        loader.setNbPoints(mesh->getVertices().size());
        for (unsigned int i=0; i<mesh->getVertices().size(); i++)
        {
            loader.addPoint(mesh->getVertices()[i][0],mesh->getVertices()[i][1],mesh->getVertices()[i][2]);
        }

        std::set< std::pair<int,int> > edges;

        const vector< vector < vector <int> > > & facets = mesh->getFacets();
        for (unsigned int i=0; i<facets.size(); i++)
        {
            const vector<int>& facet = facets[i][0];
            if (facet.size()==2)
            {
                // Line
                if (facet[0]<facet[1])
                    loader.addEdge(facet[0],facet[1]);
                else
                    loader.addEdge(facet[1],facet[0]);
            }
            else if (facet.size()==4)
            {
                // Quat
                loader.addQuad(facet[0],facet[1],facet[2],facet[3]);
            }
            else
            {
                // Triangularize
                for (unsigned int j=2; j<facet.size(); j++)
                    loader.addTriangle(facet[0],facet[j-1],facet[j]);
            }
            // Add edges
            if (facet.size()>2)
                for (unsigned int j=0; j<facet.size(); j++)
                {
                    int i1 = facet[j];
                    int i2 = facet[(j+1)%facet.size()];
                    if (edges.count(std::make_pair(i1,i2))!=0)
                    {
                        /*
                        std::cerr << "ERROR: Duplicate edge.\n";*/
                    }
                    else if (edges.count(std::make_pair(i2,i1))==0)
                    {
                        if (i1>i2)
                            loader.addEdge(i1,i2);
                        else
                            loader.addEdge(i2,i1);
                        edges.insert(std::make_pair(i1,i2));
                    }
                }
        }
        delete mesh;
    }
    else
    {
        if (!loader.load(filename))
            return false;
    }
    return true;
}


void MeshTopology::addPoint(double px, double py, double pz)
{
    seqPoints.push_back(helper::make_array(px, py, pz));
    if (seqPoints.size() > (unsigned)nbPoints)
        nbPoints = seqPoints.size();

}

void MeshTopology::addEdge( int a, int b )
{
    seqEdges.beginEdit()->push_back(Edge(a,b));
    seqEdges.endEdit();
}

void MeshTopology::addTriangle( int a, int b, int c )
{
    seqTriangles.beginEdit()->push_back( Triangle(a,b,c) );
    seqTriangles.endEdit();
}

void MeshTopology::addTetrahedron( int a, int b, int c, int d )
{
    seqTetras.push_back( Tetra(a,b,c,d) );
}

const MeshTopology::SeqEdges& MeshTopology::getEdges()
{
    if (!validEdges)
    {
        updateEdges();
        validEdges = true;
    }
    return seqEdges.getValue();
}

const MeshTopology::SeqTriangles& MeshTopology::getTriangles()
{
    if (!validTriangles)
    {
        updateTriangles();
        validTriangles = true;
    }
    return seqTriangles.getValue();
}

const MeshTopology::SeqQuads& MeshTopology::getQuads()
{
    if (!validQuads)
    {
        updateQuads();
        validQuads = true;
    }
    return seqQuads;
}

const MeshTopology::SeqTetras& MeshTopology::getTetras()
{
    if (!validTetras)
    {
        updateTetras();
        validTetras = true;
    }
    return seqTetras;
}

const MeshTopology::SeqHexas& MeshTopology::getHexas()
{
    if (!validHexas)
    {
        updateHexas();
        validHexas = true;
    }
    return seqHexas;
}

int MeshTopology::getNbPoints() const
{
    return nbPoints;
}

int MeshTopology::getNbEdges()
{
    return getEdges().size();
}

int MeshTopology::getNbTriangles()
{
    return getTriangles().size();
}

int MeshTopology::getNbQuads()
{
    return getQuads().size();
}

int MeshTopology::getNbTetras()
{
    return getTetras().size();
}

int MeshTopology::getNbHexas()
{
    return getHexas().size();
}

MeshTopology::Edge MeshTopology::getEdge(index_type i)
{
    return getEdges()[i];
}

MeshTopology::Triangle MeshTopology::getTriangle(index_type i)
{
    return getTriangles()[i];
}

MeshTopology::Quad MeshTopology::getQuad(index_type i)
{
    return getQuads()[i];
}

MeshTopology::Tetra MeshTopology::getTetra(index_type i)
{
    return getTetras()[i];
}

MeshTopology::Hexa MeshTopology::getHexa(index_type i)
{
    return getHexas()[i];
}

void MeshTopology::createEdgeVertexShellArray ()
{
    m_edgeVertexShell.resize( nbPoints );

    for (unsigned int i = 0; i < seqEdges.getValue().size(); ++i)
    {
        // adding edge i in the edge shell of both points
        m_edgeVertexShell[ seqEdges.getValue()[i][0] ].push_back( i );
        m_edgeVertexShell[ seqEdges.getValue()[i][1] ].push_back( i );
    }
}

void MeshTopology::createEdgeTriangleShellArray ()
{
    m_edgeTriangleShell.resize( getNbTriangles());
    unsigned int j;
    int edgeIndex;

    if (seqEdges.getValue().size()>0)
    {

        for (unsigned int i = 0; i < seqTriangles.getValue().size(); ++i)
        {
            const Triangle &t=seqTriangles.getValue()[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<3; ++j)
            {
                edgeIndex=getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
                assert(edgeIndex!= -1);
                m_edgeTriangleShell[i][j]=edgeIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant edges
        std::map<Edge,unsigned int> edgeMap;
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        unsigned int v1,v2;
        /// create the m_edge array at the same time than it fills the m_edgeTriangleShell array
        for (unsigned int i = 0; i < seqTriangles.getValue().size(); ++i)
        {
            const Triangle &t=seqTriangles.getValue()[i];
            for (j=0; j<3; ++j)
            {
                v1=t[(j+1)%3];
                v2=t[(j+2)%3];
                // sort vertices in lexicographics order
                if (v1<v2)
                {
                    e=Edge(v1,v2);
                }
                else
                {
                    e=Edge(v2,v1);
                }
                ite=edgeMap.find(e);
                if (ite==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    edgeIndex=edgeMap.size();
                    edgeMap[e]=edgeIndex;
                    vector<Edge> ea=seqEdges.getValue();
                    ea.push_back(e);
                    seqEdges.setValue(ea);
                }
                else
                {
                    edgeIndex=(*ite).second;
                }
                m_edgeTriangleShell[i][j]=edgeIndex;
            }
        }
    }
}

void MeshTopology::createEdgeQuadShellArray ()
{
    m_edgeQuadShell.resize( getNbQuads());
    unsigned int j;
    int edgeIndex;

    if (seqEdges.getValue().size()>0)
    {

        for (unsigned int i = 0; i < seqQuads.size(); ++i)
        {
            Quad &t=seqQuads[i];
            for (j=0; j<4; ++j)
            {
                edgeIndex=getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
                assert(edgeIndex!= -1);
                m_edgeQuadShell[i][j]=edgeIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant edges
        std::map<Edge,unsigned int> edgeMap;
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        unsigned int v1,v2;
        /// create the m_edge array at the same time than it fills the m_edgeQuadShell array
        for (unsigned int i = 0; i < seqQuads.size(); ++i)
        {
            Quad &t=seqQuads[i];
            for (j=0; j<4; ++j)
            {
                v1=t[(j+1)%4];
                v2=t[(j+2)%4];
                // sort vertices in lexicographics order
                if (v1<v2)
                {
                    e=Edge(v1,v2);
                }
                else
                {
                    e=Edge(v2,v1);
                }
                ite=edgeMap.find(e);
                if (ite==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    edgeIndex=edgeMap.size();
                    edgeMap[e]=edgeIndex;
                    vector<Edge> ea=seqEdges.getValue();
                    ea.push_back(e);
                    seqEdges.setValue(ea);
                }
                else
                {
                    edgeIndex=(*ite).second;
                }
                m_edgeQuadShell[i][j]=edgeIndex;
            }
        }
    }
}

void MeshTopology::createEdgeTetraShellArray ()
{
    m_edgeTetraShell.resize( getNbTetras());
    unsigned int j;
    int edgeIndex;
    const unsigned int tetrahedronEdgeArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    if (seqEdges.getValue().size()>0)
    {
        for (unsigned int i = 0; i < seqTetras.size(); ++i)
        {
            Tetra &t=seqTetras[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<6; ++j)
            {
                edgeIndex=getEdgeIndex(t[tetrahedronEdgeArray[j][0]],
                        t[tetrahedronEdgeArray[j][1]]);
                assert(edgeIndex!= -1);
                m_edgeTetraShell[i][j]=edgeIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant edges
        std::map<Edge,unsigned int> edgeMap;
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        unsigned int v1,v2;
        /// create the m_edge array at the same time than it fills the m_edgeTetraShell array
        for (unsigned int i = 0; i < seqTetras.size(); ++i)
        {
            Tetra &t=seqTetras[i];
            for (j=0; j<6; ++j)
            {
                v1=t[tetrahedronEdgeArray[j][0]];
                v2=t[tetrahedronEdgeArray[j][1]];
                // sort vertices in lexicographics order
                if (v1<v2)
                {
                    e=Edge(v1,v2);
                }
                else
                {
                    e=Edge(v2,v1);
                }
                ite=edgeMap.find(e);
                if (ite==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    edgeIndex=edgeMap.size();
                    edgeMap[e]=edgeIndex;
                    vector<Edge> ea=seqEdges.getValue();
                    ea.push_back(e);
                    seqEdges.setValue(ea);
                }
                else
                {
                    edgeIndex=(*ite).second;
                }
                m_edgeTetraShell[i][j]=edgeIndex;
            }
        }
    }
}

void MeshTopology::createEdgeHexaShellArray ()
{
    m_edgeHexaShell.resize( getNbHexas());
    unsigned int j;
    int edgeIndex;
    unsigned int edgeHexahedronDescriptionArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};

    if (seqEdges.getValue().size()>0)
    {
        for (unsigned int i = 0; i < m_edgeHexaShell.size(); ++i)
        {
            Hexa &h=seqHexas[i];
            // adding edge i in the edge shell of both points
            for (j=0; j<12; ++j)
            {
                edgeIndex=getEdgeIndex(h[edgeHexahedronDescriptionArray[j][0]],
                        h[edgeHexahedronDescriptionArray[j][1]]);
                assert(edgeIndex!= -1);
                m_edgeHexaShell[i][j]=edgeIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant edges
        std::map<Edge,unsigned int> edgeMap;
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        unsigned int v1,v2;
        /// create the m_edge array at the same time than it fills the m_hexahedronEdge array
        for (unsigned int i = 0; i < m_edgeHexaShell.size(); ++i)
        {
            Hexa &h=seqHexas[i];
            for (j=0; j<12; ++j)
            {
                v1=h[edgeHexahedronDescriptionArray[j][0]];
                v2=h[edgeHexahedronDescriptionArray[j][1]];
                // sort vertices in lexicographics order
                if (v1<v2)
                {
                    e=Edge(v1,v2);
                }
                else
                {
                    e=Edge(v2,v1);
                }
                ite=edgeMap.find(e);
                if (ite==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    edgeIndex=edgeMap.size();
                    edgeMap[e]=edgeIndex;
                    vector<Edge> ea=seqEdges.getValue();
                    ea.push_back(e);
                    seqEdges.setValue(ea);
                }
                else
                {
                    edgeIndex=(*ite).second;
                }
                m_edgeHexaShell[i][j]=edgeIndex;
            }
        }
    }
}

void MeshTopology::createTriangleVertexShellArray ()
{
    m_triangleVertexShell.resize( nbPoints );
    unsigned int j;

    for (unsigned int i = 0; i < seqTriangles.getValue().size(); ++i)
    {
        // adding triangle i in the triangle shell of all points
        for (j=0; j<3; ++j)
            m_triangleVertexShell[ seqTriangles.getValue()[i][j]  ].push_back( i );
    }
}

void MeshTopology::createTriangleEdgeShellArray ()
{
    if (m_edgeTriangleShell.empty())
        createEdgeTriangleShellArray();
    m_triangleEdgeShell.resize( getNbEdges());
    const vector< TriangleEdges > &tea=m_edgeTriangleShell;
    unsigned int j;

    for (unsigned int i = 0; i < seqTriangles.getValue().size(); ++i)
    {
        // adding triangle i in the triangle shell of all edges
        for (j=0; j<3; ++j)
        {
            m_triangleEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}

void MeshTopology::createTriangleTetraShellArray ()
{
    m_triangleTetraShell.resize( getNbTetras());
    unsigned int j;
    int triangleIndex;

    if (seqTriangles.getValue().size()>0)
    {
        for (unsigned int i = 0; i < seqTetras.size(); ++i)
        {
            Tetra &t=seqTetras[i];
            // adding triangles in the triangle list of the ith tetrahedron  i
            for (j=0; j<4; ++j)
            {
                triangleIndex=getTriangleIndex(t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
                assert(triangleIndex!= -1);
                m_triangleTetraShell[i][j]=triangleIndex;
            }
        }
    }
    else
    {
        // create a temporary map to find redundant triangles
        std::map<Triangle,unsigned int> triangleMap;
        std::map<Triangle,unsigned int>::iterator itt;
        Triangle tr;
        unsigned int v[3],val;
        /// create the m_edge array at the same time than it fills the m_triangleTetraShell array
        for (unsigned int i = 0; i < seqTetras.size(); ++i)
        {
            Tetra &t=seqTetras[i];
            for (j=0; j<4; ++j)
            {
                if (j%2)
                {
                    v[0]=t[(j+1)%4]; v[1]=t[(j+2)%4]; v[2]=t[(j+3)%4];
                }
                else
                {
                    v[0]=t[(j+1)%4]; v[2]=t[(j+2)%4]; v[1]=t[(j+3)%4];
                }
                //		std::sort(v,v+2);
                // sort v such that v[0] is the smallest one
                while ((v[0]>v[1]) || (v[0]>v[2]))
                {
                    val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=val;
                }
                // check if a triangle with an opposite orientation already exists
                tr=helper::make_array<unsigned int>(v[0],v[2],v[1]);
                itt=triangleMap.find(tr);
                if (itt==triangleMap.end())
                {
                    // edge not in edgeMap so create a new one
                    triangleIndex=triangleMap.size();
                    tr=helper::make_array<unsigned int>(v[0],v[1],v[2]);
                    triangleMap[tr]=triangleIndex;
                    vector<Triangle> ta=seqTriangles.getValue();
                    ta.push_back(tr);
                    seqTriangles.setValue(ta);
                }
                else
                {
                    triangleIndex=(*itt).second;
                }
                m_triangleTetraShell[i][j]=triangleIndex;
            }
        }
    }
}


void MeshTopology::createQuadVertexShellArray ()
{
    m_quadVertexShell.resize( nbPoints );
    unsigned int j;

    for (unsigned int i = 0; i < seqQuads.size(); ++i)
    {
        // adding quad i in the quad shell of all points
        for (j=0; j<4; ++j)
            m_quadVertexShell[ seqQuads[i][j]  ].push_back( i );
    }
}

void MeshTopology::createQuadEdgeShellArray ()
{
    if (m_edgeQuadShell.empty())
        createEdgeQuadShellArray();
    m_quadEdgeShell.resize( getNbEdges() );
    unsigned int j;
    for (unsigned int i = 0; i < seqQuads.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
        {
            m_quadEdgeShell[ m_edgeQuadShell[i][j] ].push_back( i );
        }
    }
}

void MeshTopology::createQuadHexaShellArray ()
{
    m_quadHexaShell.resize(getNbHexas());
    int quadIndex;

    if (seqQuads.size()>0)
    {
        for (unsigned int i = 0; i < seqHexas.size(); ++i)
        {
            Hexa &h=seqHexas[i];
            // adding the 6 quads in the quad list of the ith hexahedron  i
            // Quad 0 :
            quadIndex=getQuadIndex(h[0],h[3],h[2],h[1]);
            assert(quadIndex!= -1);
            m_quadHexaShell[i][0]=quadIndex;
            // Quad 1 :
            quadIndex=getQuadIndex(h[4],h[5],h[6],h[7]);
            assert(quadIndex!= -1);
            m_quadHexaShell[i][1]=quadIndex;
            // Quad 2 :
            quadIndex=getQuadIndex(h[0],h[1],h[5],h[4]);
            assert(quadIndex!= -1);
            m_quadHexaShell[i][2]=quadIndex;
            // Quad 3 :
            quadIndex=getQuadIndex(h[1],h[2],h[6],h[5]);
            assert(quadIndex!= -1);
            m_quadHexaShell[i][3]=quadIndex;
            // Quad 4 :
            quadIndex=getQuadIndex(h[2],h[3],h[7],h[6]);
            assert(quadIndex!= -1);
            m_quadHexaShell[i][4]=quadIndex;
            // Quad 5 :
            quadIndex=getQuadIndex(h[3],h[0],h[4],h[7]);
            assert(quadIndex!= -1);
            m_quadHexaShell[i][5]=quadIndex;
        }
    }
    else
    {
        // create a temporary map to find redundant quads
        std::map<Quad,unsigned int> quadMap;
        std::map<Quad,unsigned int>::iterator itt;
        Quad qu;
        unsigned int v[4],val;
        /// create the m_edge array at the same time than it fills the m_hexahedronEdge array
        for (unsigned int i = 0; i < seqHexas.size(); ++i)
        {
            Hexa &h=seqHexas[i];

            // Quad 0 :
            v[0]=h[0]; v[1]=h[3]; v[2]=h[2]; v[3]=h[1];
            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
            {
                val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=v[3]; v[3]=val;
            }
            //std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            //std::sort(v+1,v+2); std::sort(v+1,v+3);
            //std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=seqQuads.size();
                quadMap[qu]=quadIndex;
                qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
                quadMap[qu]=quadIndex;
                seqQuads.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_quadHexaShell[i][0]=quadIndex;

            // Quad 1 :
            v[0]=h[4]; v[1]=h[5]; v[2]=h[6]; v[3]=h[7];
            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
            {
                val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=v[3]; v[3]=val;
            }
            //std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            //std::sort(v+1,v+2); std::sort(v+1,v+3);
            //std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=seqQuads.size();
                quadMap[qu]=quadIndex;
                qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
                quadMap[qu]=quadIndex;
                seqQuads.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_quadHexaShell[i][1]=quadIndex;

            // Quad 2 :
            v[0]=h[0]; v[1]=h[1]; v[2]=h[5]; v[3]=h[4];
            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
            {
                val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=v[3]; v[3]=val;
            }
            //std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            //std::sort(v+1,v+2); std::sort(v+1,v+3);
            //std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=seqQuads.size();
                quadMap[qu]=quadIndex;
                qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
                quadMap[qu]=quadIndex;
                seqQuads.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_quadHexaShell[i][2]=quadIndex;

            // Quad 3 :
            v[0]=h[1]; v[1]=h[2]; v[2]=h[6]; v[3]=h[5];
            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
            {
                val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=v[3]; v[3]=val;
            }
            //std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            //std::sort(v+1,v+2); std::sort(v+1,v+3);
            //std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=seqQuads.size();
                quadMap[qu]=quadIndex;
                qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
                quadMap[qu]=quadIndex;
                seqQuads.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_quadHexaShell[i][3]=quadIndex;

            // Quad 4 :
            v[0]=h[2]; v[1]=h[3]; v[2]=h[7]; v[3]=h[6];
            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
            {
                val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=v[3]; v[3]=val;
            }
            //std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            //std::sort(v+1,v+2); std::sort(v+1,v+3);
            //std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=seqQuads.size();
                quadMap[qu]=quadIndex;
                qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
                quadMap[qu]=quadIndex;
                seqQuads.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_quadHexaShell[i][4]=quadIndex;

            // Quad 5 :
            v[0]=h[3]; v[1]=h[0]; v[2]=h[4]; v[3]=h[7];
            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]) || (v[0]>v[3]))
            {
                val=v[0]; v[0]=v[1]; v[1]=v[2]; v[2]=v[3]; v[3]=val;
            }
            //std::sort(v,v+1); std::sort(v,v+2); std::sort(v,v+3);
            //std::sort(v+1,v+2); std::sort(v+1,v+3);
            //std::sort(v+2,v+3);
            // sort vertices in lexicographics order
            qu=helper::make_array<unsigned int>(v[0],v[3],v[2],v[1]);
            itt=quadMap.find(qu);
            if (itt==quadMap.end())
            {
                // quad not in edgeMap so create a new one
                quadIndex=seqQuads.size();
                quadMap[qu]=quadIndex;
                qu=helper::make_array<unsigned int>(v[0],v[1],v[2],v[3]);
                quadMap[qu]=quadIndex;
                seqQuads.push_back(qu);
            }
            else
            {
                quadIndex=(*itt).second;
            }
            m_quadHexaShell[i][5]=quadIndex;
        }
    }
}

void MeshTopology::createTetraVertexShellArray ()
{
    m_tetraVertexShell.resize( getNbPoints() );
    unsigned int j;

    for (unsigned int i = 0; i < seqTetras.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
            m_tetraVertexShell[ seqTetras[i][j]  ].push_back( i );
    }
}

void MeshTopology::createTetraEdgeShellArray ()
{
    if (!m_edgeTetraShell.size())
        createEdgeTetraShellArray();
    m_tetraEdgeShell.resize( getNbEdges() );
    const vector< TetraEdges > &tea = m_edgeTetraShell;
    unsigned int j;

    for (unsigned int i = 0; i < seqTetras.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<6; ++j)
        {
            m_tetraEdgeShell[ tea[i][j] ].push_back( i );
        }
    }
}

void MeshTopology::createTetraTriangleShellArray ()
{
    if (!m_triangleTetraShell.size())
        createTriangleTetraShellArray();
    m_tetraTriangleShell.resize( getNbTriangles());
    unsigned int j;
    const vector< TetraTriangles > &tta=m_triangleTetraShell;


    for (unsigned int i = 0; i < seqTetras.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (j=0; j<4; ++j)
        {
            m_tetraTriangleShell[ tta[i][j] ].push_back( i );
        }
    }
}

void MeshTopology::createHexaVertexShellArray ()
{
    m_hexaVertexShell.resize( getNbPoints() );
    unsigned int j;

    for (unsigned int i = 0; i < seqHexas.size(); ++i)
    {
        // adding vertex i in the vertex shell
        for (j=0; j<8; ++j)
            m_hexaVertexShell[ seqHexas[i][j]  ].push_back( i );
    }
}

void MeshTopology::createHexaEdgeShellArray ()
{
    if (!m_edgeHexaShell.size())
        createEdgeHexaShellArray();
    m_hexaEdgeShell.resize(getNbEdges());
    unsigned int j;
    const vector< HexaEdges > &hea=m_edgeHexaShell;


    for (unsigned int i = 0; i < seqHexas.size(); ++i)
    {
        // adding edge i in the edge shell
        for (j=0; j<12; ++j)
        {
            m_hexaEdgeShell[ hea[i][j] ].push_back( i );
        }
    }
}

void MeshTopology::createHexaQuadShellArray ()
{
    if (!m_quadHexaShell.size())
        createQuadHexaShellArray();
    m_hexaQuadShell.resize( getNbQuads());
    unsigned int j;
    const vector< HexaQuads > &qha=m_quadHexaShell;


    for (unsigned int i = 0; i < seqHexas.size(); ++i)
    {
        // adding quad i in the edge shell of both points
        for (j=0; j<6; ++j)
        {
            m_hexaQuadShell[ qha[i][j] ].push_back( i );
        }
    }
}

const MeshTopology::VertexEdges& MeshTopology::getEdgeVertexShell(PointID i)
{
    if (!m_edgeVertexShell.size())
        createEdgeVertexShellArray();
    return m_edgeVertexShell[i];
}

const MeshTopology::TriangleEdges& MeshTopology::getEdgeTriangleShell(TriangleID i)
{
    if (m_edgeTriangleShell.empty())
        createEdgeTriangleShellArray();
    return m_edgeTriangleShell[i];
}

const MeshTopology::QuadEdges& MeshTopology::getEdgeQuadShell(QuadID i)
{
    if (m_edgeQuadShell.empty())
        createEdgeQuadShellArray();
    return m_edgeQuadShell[i];
}

const MeshTopology::TetraEdges& MeshTopology::getEdgeTetraShell(TetraID i)
{
    if (!m_edgeTetraShell.empty())
        createEdgeTetraShellArray();
    return m_edgeTetraShell[i];
}

const MeshTopology::HexaEdges& MeshTopology::getEdgeHexaShell(HexaID i)
{
    if (!m_edgeHexaShell.size())
        createEdgeHexaShellArray();
    return m_edgeHexaShell[i];
}

const MeshTopology::VertexTriangles& MeshTopology::getTriangleVertexShell(PointID i)
{
    if (!m_triangleVertexShell.size())
        createTriangleVertexShellArray();
    return m_triangleVertexShell[i];
}

const MeshTopology::EdgeTriangles& MeshTopology::getTriangleEdgeShell(EdgeID i)
{
    if (m_triangleEdgeShell.empty())
        createTriangleEdgeShellArray();
    return m_triangleEdgeShell[i];
}

const MeshTopology::TetraTriangles& MeshTopology::getTriangleTetraShell(TetraID i)
{
    if (!m_triangleTetraShell.size())
        createTriangleTetraShellArray();
    return m_triangleTetraShell[i];
}

const MeshTopology::VertexQuads& MeshTopology::getQuadVertexShell(PointID i)
{
    if (m_quadVertexShell.empty())
        createQuadVertexShellArray();
    return m_quadVertexShell[i];
}

const vector< MeshTopology::QuadID >& MeshTopology::getQuadEdgeShell(EdgeID i)
{
    if (!m_quadEdgeShell.size())
        createQuadEdgeShellArray();
    return m_quadEdgeShell[i];
}

const MeshTopology::HexaQuads& MeshTopology::getQuadHexaShell(HexaID i)
{
    if (!m_quadHexaShell.size())
        createQuadHexaShellArray();
    return m_quadHexaShell[i];
}

const MeshTopology::VertexTetras& MeshTopology::getTetraVertexShell(PointID i)
{
    if (!m_tetraVertexShell.size())
        createTetraVertexShellArray();
    return m_tetraVertexShell[i];
}

const MeshTopology::EdgeTetras& MeshTopology::getTetraEdgeShell(EdgeID i)
{
    if (!m_tetraEdgeShell.size())
        createTetraEdgeShellArray();
    return m_tetraEdgeShell[i];
}

const MeshTopology::TriangleTetras& MeshTopology::getTetraTriangleShell(TriangleID i)
{
    if (!m_tetraTriangleShell.size())
        createTetraTriangleShellArray();
    return m_tetraTriangleShell[i];
}

const MeshTopology::VertexHexas& MeshTopology::getHexaVertexShell(PointID i)
{
    if (!m_hexaVertexShell.size())
        createHexaVertexShellArray();
    return m_hexaVertexShell[i];
}

const MeshTopology::EdgeHexas& MeshTopology::getHexaEdgeShell(EdgeID i)
{
    if (!m_hexaEdgeShell.size())
        createHexaEdgeShellArray();
    return m_hexaEdgeShell[i];
}

const MeshTopology::QuadHexas& MeshTopology::getHexaQuadShell(QuadID i)
{
    if (!m_hexaQuadShell.size())
        createHexaQuadShellArray();
    return m_hexaQuadShell[i];
}

const vector< MeshTopology::VertexTriangles >& MeshTopology::getTriangleVertexShellArray()
{
    if (!m_triangleVertexShell.size())
        createTriangleVertexShellArray();
    return m_triangleVertexShell;
}

const vector< MeshTopology::VertexQuads >& MeshTopology::getQuadVertexShellArray()
{
    if (!m_quadVertexShell.size())
        createQuadVertexShellArray();
    return m_quadVertexShell;
}

int MeshTopology::getEdgeIndex(PointID v1, PointID v2)
{
    const VertexEdges &es1 = getEdgeVertexShell(v1) ;
    const SeqEdges &ea = getEdges();
    unsigned int i=0;
    int result= -1;
    while ((i<es1.size()) && (result== -1))
    {
        const MeshTopology::Edge &e=ea[es1[i]];
        if ((e[0]==v2)|| (e[1]==v2))
            result=(int) es1[i];

        i++;
    }
    return result;
}

int MeshTopology::getTriangleIndex(PointID v1, PointID v2, PointID v3)
{
    const vector< VertexTriangles > &tvs=getTriangleVertexShellArray();

    const vector<TriangleID> &set1=tvs[v1];
    const vector<TriangleID> &set2=tvs[v2];
    const vector<TriangleID> &set3=tvs[v3];

    // The destination vector must be large enough to contain the result.
    vector<TriangleID> out1(set1.size()+set2.size());
    vector<TriangleID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    vector<TriangleID> out2(set3.size()+out1.size());
    vector<TriangleID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    assert(out2.size()==0 || out2.size()==1);

    if (out2.size()==1)
        return (int) (out2[0]);
    else
        return -1;
}

int MeshTopology::getQuadIndex(PointID v1, PointID v2, PointID v3,  PointID v4)
{
    const vector< VertexQuads > &qvs=getQuadVertexShellArray();

    const vector<QuadID> &set1=qvs[v1];
    const vector<QuadID> &set2=qvs[v2];
    const vector<QuadID> &set3=qvs[v3];
    const vector<QuadID> &set4=qvs[v4];

    // The destination vector must be large enough to contain the result.
    vector<QuadID> out1(set1.size()+set2.size());
    vector<QuadID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    vector<QuadID> out2(set3.size()+out1.size());
    vector<QuadID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    vector<QuadID> out3(set4.size()+out2.size());
    vector<QuadID>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

    assert(out3.size()==0 || out3.size()==1);

    if (out3.size()==1)
        return (int) (out3[0]);
    else
        return -1;
}

bool MeshTopology::hasPos() const
{
    return !seqPoints.empty();
}

double MeshTopology::getPX(int i) const
{
    return ((unsigned)i<seqPoints.size()?seqPoints[i][0]:0.0);
}

double MeshTopology::getPY(int i) const
{
    return ((unsigned)i<seqPoints.size()?seqPoints[i][1]:0.0);
}

double MeshTopology::getPZ(int i) const
{
    return ((unsigned)i<seqPoints.size()?seqPoints[i][2]:0.0);
}

void MeshTopology::invalidate()
{
    validEdges = false;
    validTriangles = false;
    validQuads = false;
    validTetras = false;
    validHexas = false;
    ++revision;
    //std::cout << "MeshTopology::invalidate()"<<std::endl;
}

} // namespace topology

} // namespace component

} // namespace sofa

