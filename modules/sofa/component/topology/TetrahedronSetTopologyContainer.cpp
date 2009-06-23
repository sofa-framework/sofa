/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <sofa/component/topology/TetrahedronSetTopologyContainer.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/component/container/MeshLoader.h>

namespace sofa
{

namespace component
{

namespace topology
{

using namespace std;
using namespace sofa::defaulttype;

SOFA_DECL_CLASS(TetrahedronSetTopologyContainer)
int TetrahedronSetTopologyContainerClass = core::RegisterObject("Tetrahedron set topology container")
        .add< TetrahedronSetTopologyContainer >()
        ;

const unsigned int tetrahedronEdgeArray[6][2] = {{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}};

TetrahedronSetTopologyContainer::TetrahedronSetTopologyContainer()
    : TriangleSetTopologyContainer()
    , d_tetrahedron(initDataPtr(&d_tetrahedron, &m_tetrahedron, "tetras", "List of tetrahedron indices"))
{
}


TetrahedronSetTopologyContainer::TetrahedronSetTopologyContainer(const sofa::helper::vector< Tetrahedron >& tetrahedra )
    : TriangleSetTopologyContainer()
    , m_tetrahedron( tetrahedra )
    , d_tetrahedron(initDataPtr(&d_tetrahedron, &m_tetrahedron, "tetras", "List of tetrahedron indices"))
{
    for (unsigned int i=0; i<m_tetrahedron.size(); ++i)
    {
        for(unsigned int j=0; j<4; ++j)
        {
            int a = m_tetrahedron[i][j];
            if (a >= getNbPoints()) nbPoints.setValue(a+1);
        }
    }
}


void TetrahedronSetTopologyContainer::addTetra( int a, int b, int c, int d )
{
    d_tetrahedron.beginEdit();
    m_tetrahedron.push_back(Tetra(a,b,c,d));
    d_tetrahedron.endEdit();
    if (a >= getNbPoints()) nbPoints.setValue(a+1);
    if (b >= getNbPoints()) nbPoints.setValue(b+1);
    if (c >= getNbPoints()) nbPoints.setValue(c+1);
    if (d >= getNbPoints()) nbPoints.setValue(d+1);
}

void TetrahedronSetTopologyContainer::init()
{
    d_tetrahedron.getValue(); // make sure m_tetrahedron is up to date
    if (!m_tetrahedron.empty())
    {
        for (unsigned int i=0; i<m_tetrahedron.size(); ++i)
        {
            for(unsigned int j=0; j<4; ++j)
            {
                int a = m_tetrahedron[i][j];
                if (a >= getNbPoints()) nbPoints.setValue(a+1);
            }
        }
    }

    TriangleSetTopologyContainer::init();
}

void TetrahedronSetTopologyContainer::loadFromMeshLoader(sofa::component::container::MeshLoader* loader)
{
    // load points
    PointSetTopologyContainer::loadFromMeshLoader(loader);
    d_tetrahedron.beginEdit();
    loader->getTetras(m_tetrahedron);
    d_tetrahedron.endEdit();
}

void TetrahedronSetTopologyContainer::createTetrahedronSetArray()
{
#ifndef NDEBUG
    sout << "Error. [TetrahedronSetTopologyContainer::createTetrahedronSetArray] This method must be implemented by a child topology." << endl;
#endif
}

void TetrahedronSetTopologyContainer::createEdgeSetArray()
{
    d_edge.beginEdit();
    if(hasEdges())
    {
#ifndef NDEBUG
        sout << "Warning. [TetrahedronSetTopologyContainer::createEdgeSetArray] edge array is not empty." << endl;
#endif

        EdgeSetTopologyContainer::clear();

        clearTriangleEdges();
        clearTriangleEdgeShell();

        clearTetrahedronEdges();
        clearTetrahedronEdgeShell();
    }

    // create a temporary map to find redundant edges
    std::map<Edge,unsigned int> edgeMap;

    /// create the m_edge array at the same time than it fills the m_tetrahedronEdge array
    for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        const Tetrahedron &t = m_tetrahedron[i];
        for (unsigned int j=0; j<6; ++j)
        {
            const unsigned int v1 = t[(j+1)%6];
            const unsigned int v2 = t[(j+2)%6];

            // sort vertices in lexicographic order
            const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

            if (edgeMap.find(e)==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                const int edgeIndex = edgeMap.size();
                edgeMap[e] = edgeIndex;
                m_edge.push_back(e);
            }
        }
    }
    d_edge.endEdit();
}

void TetrahedronSetTopologyContainer::createTetrahedronEdgeArray()
{
    if(!hasTetrahedra()) // this method should only be called when triangles exist
    {
#ifndef NDEBUG
        sout << "Warning. [TetrahedronSetTopologyContainer::createTetrahedronEdgeArray] tetra array is empty." << endl;
#endif
        createTetrahedronSetArray();
    }

    if(hasTetrahedronEdges())
        clearTetrahedronEdges();

    if(!hasEdges()) // To optimize, this method should be called without creating edgesArray before.
    {
#ifndef NDEBUG
        sout << "Warning. [TetrahedronSetTopologyContainer::createTetrahedronEdgeArray] edge array is empty." << endl;
#endif

        /// create edge array and triangle edge array at the same time
        const unsigned int numTetra = getNumberOfTetrahedra();
        m_tetrahedronEdge.resize (numTetra);

        d_edge.beginEdit();
        // create a temporary map to find redundant edges
        std::map<Edge,unsigned int> edgeMap;

        /// create the m_edge array at the same time than it fills the m_tetrahedronEdge array
        for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
        {
            const Tetrahedron &t = m_tetrahedron[i];
            for (unsigned int j=0; j<6; ++j)
            {
                const unsigned int v1 = t[(j+1)%6];
                const unsigned int v2 = t[(j+2)%6];

                // sort vertices in lexicographic order
                const Edge e((v1<v2) ? Edge(v1,v2) : Edge(v2,v1));

                if (edgeMap.find(e)==edgeMap.end())
                {
                    // edge not in edgeMap so create a new one
                    const int edgeIndex = edgeMap.size();
                    edgeMap[e] = edgeIndex;
                    m_edge.push_back(e);
                }
                m_tetrahedronEdge[i][j] = edgeMap[e];
            }
        }
        d_edge.endEdit();
    }
    else
    {
        /// there are already existing edges : must use an inefficient method. Parse all triangles and find the edge that match each triangle edge
        const unsigned int numTetra = getNumberOfTetrahedra();
        const unsigned int numEdges = getNumberOfEdges();

        m_tetrahedronEdge.resize(numTetra);
        /// create a multi map where the key is a vertex index and the content is the indices of edges adjacent to that vertex.
        std::multimap<PointID, EdgeID> edgeVertexShellMap;
        std::multimap<PointID, EdgeID>::iterator it;
        bool foundEdge;

        for (unsigned int edge=0; edge<numEdges; ++edge)  //Todo: check if not better using multimap <PointID ,TriangleID> and for each edge, push each triangle present in both shell
        {
            edgeVertexShellMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][0],edge));
            edgeVertexShellMap.insert(std::pair<PointID, EdgeID> (m_edge[edge][1],edge));
        }

        for(unsigned int i=0; i<numTetra; ++i)
        {
            Tetrahedron &t = m_tetrahedron[i];
            // adding edge i in the edge shell of both points
            for(unsigned int j=0; j<6; ++j)
            {
                //finding edge i in edge array
                std::pair<std::multimap<PointID, EdgeID>::iterator, std::multimap<PointID, EdgeID>::iterator > itPair=edgeVertexShellMap.equal_range(t[(j+1)%6]);

                foundEdge=false;
                for(it=itPair.first; (it!=itPair.second) && (foundEdge==false); ++it)
                {
                    unsigned int edge = (*it).second;
                    if ( (m_edge[edge][0] == t[(j+1)%6] && m_edge[edge][1] == t[(j+2)%6]) || (m_edge[edge][0] == t[(j+2)%6] && m_edge[edge][1] == t[(j+1)%6]))
                    {
                        m_tetrahedronEdge[i][j] = edge;
                        foundEdge=true;
                    }
                }
#ifndef NDEBUG
                if (foundEdge==false)
                    sout << "[TetrahedronSetTopologyContainer::getTetrahedronArray] cannot find edge for tetrahedron " << i << "and edge "<< j << endl;
#endif
            }
        }
    }
}

void TetrahedronSetTopologyContainer::createTriangleSetArray()
{
    d_triangle.beginEdit();
    if(hasTriangles())
    {
        TriangleSetTopologyContainer::clear();
        clearTriangles();
        clearTetrahedronTriangles();
        clearTetrahedronTriangleShell();
    }

    // create a temporary map to find redundant triangles
    std::map<Triangle,unsigned int> triangleMap;

    /// create the m_edge array at the same time than it fills the m_tetrahedronEdge array
    for (unsigned int i=0; i<m_tetrahedron.size(); ++i)
    {
        Tetrahedron &t = m_tetrahedron[i];

        for (unsigned int j=0; j<4; ++j)
        {
            unsigned int v[3];

            if (j%2)
            {
                v[0]=t[(j+1)%4];
                v[1]=t[(j+2)%4];
                v[2]=t[(j+3)%4];
            }
            else
            {
                v[0]=t[(j+1)%4];
                v[2]=t[(j+2)%4];
                v[1]=t[(j+3)%4];
            }

            // sort v such that v[0] is the smallest one
            while ((v[0]>v[1]) || (v[0]>v[2]))
            {
                unsigned int val=v[0];
                v[0]=v[1];
                v[1]=v[2];
                v[2]=val;
            }

            // check if a triangle with an opposite orientation already exists
            Triangle tr = helper::make_array<unsigned int>(v[0], v[2], v[1]);

            if (triangleMap.find(tr) == triangleMap.end())
            {
                // triangle not in triangleMap so create a new one
                tr = helper::make_array<unsigned int>(v[0], v[1], v[2]);
                if (triangleMap.find(tr) == triangleMap.end())
                {
                    triangleMap[tr] = m_triangle.size();
                    m_triangle.push_back(tr);
                }
                else
                {
                    serr << "ERROR: duplicate triangle " << tr << " in tetra " << i <<" : " << t << sendl;
                }
            }
        }
    }
    d_triangle.endEdit();
}

void TetrahedronSetTopologyContainer::createTetrahedronTriangleArray()
{
    if(!hasTriangles())
        createTriangleSetArray();

    if(hasTetrahedronTriangles())
        clearTetrahedronTriangles();

    m_tetrahedronTriangle.resize( getNumberOfTetrahedra());

    for(unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        Tetrahedron &t=m_tetrahedron[i];

        // adding triangles in the triangle list of the ith tetrahedron  i
        for (unsigned int j=0; j<4; ++j)
        {
            const int triangleIndex = getTriangleIndex(t[(j+1)%4], t[(j+2)%4], t[(j+3)%4]);
            m_tetrahedronTriangle[i][j] = (unsigned int) triangleIndex;
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedronVertexShellArray ()
{
    if(hasTetrahedronVertexShell())
        clearTetrahedronVertexShell();

    m_tetrahedronVertexShell.resize( getNbPoints() );

    for (unsigned int i = 0; i < m_tetrahedron.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<4; ++j)
        {
            m_tetrahedronVertexShell[ m_tetrahedron[i][j]  ].push_back( i );
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedronEdgeShellArray ()
{
    if(!hasTetrahedronEdges())
        createTetrahedronEdgeArray();

    if(hasTetrahedronEdgeShell())
        clearTetrahedronEdgeShell();

    m_tetrahedronEdgeShell.resize(getNumberOfEdges());

    for (unsigned int i=0; i<m_tetrahedron.size(); ++i)
    {
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<6; ++j)
        {
            m_tetrahedronEdgeShell[ m_tetrahedronEdge[i][j] ].push_back( i );
        }
    }
}

void TetrahedronSetTopologyContainer::createTetrahedronTriangleShellArray ()
{
    if(!hasTetrahedronTriangles())
        createTetrahedronTriangleArray();

    if(hasTetrahedronTriangleShell())
        clearTetrahedronTriangleShell();

    m_tetrahedronTriangleShell.resize( getNumberOfTriangles());

    for (unsigned int i=0; i<m_tetrahedron.size(); ++i)
    {
        // adding tetrahedron i in the shell of all neighbors triangles
        for (unsigned int j=0; j<4; ++j)
        {
            m_tetrahedronTriangleShell[ m_tetrahedronTriangle[i][j] ].push_back( i );
        }
    }
}

const sofa::helper::vector<Tetrahedron> &TetrahedronSetTopologyContainer::getTetrahedronArray()
{
    if (!hasTetrahedra() && getNbPoints()>0)
    {
#ifndef NDEBUG
        sout << "[TetrahedronSetTopologyContainer::getTetrahedronArray] creating tetrahedron array." << endl;
#endif
        createTetrahedronSetArray();
    }

    return m_tetrahedron;
}

int TetrahedronSetTopologyContainer::getTetrahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4)
{
    if(!hasTetrahedronVertexShell())
        createTetrahedronVertexShellArray();

    sofa::helper::vector<unsigned int> set1 = getTetrahedronVertexShell(v1);
    sofa::helper::vector<unsigned int> set2 = getTetrahedronVertexShell(v2);
    sofa::helper::vector<unsigned int> set3 = getTetrahedronVertexShell(v3);
    sofa::helper::vector<unsigned int> set4 = getTetrahedronVertexShell(v4);

    sort(set1.begin(), set1.end());
    sort(set2.begin(), set2.end());
    sort(set3.begin(), set3.end());
    sort(set4.begin(), set4.end());

    // The destination vector must be large enough to contain the result.
    sofa::helper::vector<unsigned int> out1(set1.size()+set2.size());
    sofa::helper::vector<unsigned int>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    sofa::helper::vector<unsigned int> out2(set3.size()+out1.size());
    sofa::helper::vector<unsigned int>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    sofa::helper::vector<unsigned int> out3(set4.size()+out2.size());
    sofa::helper::vector<unsigned int>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

    assert(out3.size()==0 || out3.size()==1);

    if (out3.size()==1)
        return (int) (out3[0]);
    else
        return -1;
}

unsigned int TetrahedronSetTopologyContainer::getNumberOfTetrahedra() const
{
    return m_tetrahedron.size();
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronVertexShellArray()
{
    if (!hasTetrahedronVertexShell())
        createTetrahedronVertexShellArray();

    return m_tetrahedronVertexShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShellArray()
{
    if (!hasTetrahedronEdgeShell())
        createTetrahedronEdgeShellArray();

    return m_tetrahedronEdgeShell;
}

const sofa::helper::vector< sofa::helper::vector<unsigned int> > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShellArray()
{
    if (!hasTetrahedronTriangleShell())
        createTetrahedronTriangleShellArray();

    return m_tetrahedronTriangleShell;
}

const sofa::helper::vector< TetrahedronEdges> &TetrahedronSetTopologyContainer::getTetrahedronEdgeArray()
{
    if (!hasTetrahedronEdges())
        createTetrahedronEdgeArray();

    return m_tetrahedronEdge;
}

Edge TetrahedronSetTopologyContainer::getLocalTetrahedronEdges (const unsigned int i) const
{
    assert(i<6);
    return Edge (tetrahedronEdgeArray[i][0], tetrahedronEdgeArray[i][1]);
}

const sofa::helper::vector< TetrahedronTriangles> &TetrahedronSetTopologyContainer::getTetrahedronTriangleArray()
{
    if (!hasTetrahedronTriangles())
        createTetrahedronTriangleArray();

    return m_tetrahedronTriangle;
}

const sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronVertexShell(const unsigned int i)
{
    if (!hasTetrahedronVertexShell())
        createTetrahedronVertexShellArray();

    assert(i < m_tetrahedronVertexShell.size());

    return m_tetrahedronVertexShell[i];
}

const sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShell(const unsigned int i)
{
    if (!hasTetrahedronEdgeShell())
        createTetrahedronEdgeShellArray();

    assert(i < m_tetrahedronEdgeShell.size());

    return m_tetrahedronEdgeShell[i];
}

const sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShell(const unsigned int i)
{
    if (!hasTetrahedronTriangleShell())
        createTetrahedronTriangleShellArray();

    assert(i < m_tetrahedronTriangleShell.size());

    return m_tetrahedronTriangleShell[i];
}

const TetrahedronEdges &TetrahedronSetTopologyContainer::getTetrahedronEdges(const unsigned int i)
{
    if (!hasTetrahedronEdges())
        createTetrahedronEdgeArray();

    assert(i < m_tetrahedronEdge.size());

    return m_tetrahedronEdge[i];
}

const TetrahedronTriangles &TetrahedronSetTopologyContainer::getTetrahedronTriangles(const unsigned int i)
{
    if (!hasTetrahedronTriangles())
        createTetrahedronTriangleArray();

    assert(i < m_tetrahedronTriangle.size());

    return m_tetrahedronTriangle[i];
}

int TetrahedronSetTopologyContainer::getVertexIndexInTetrahedron(const Tetrahedron &t,
        unsigned int vertexIndex) const
{
    if (t[0]==vertexIndex)
        return 0;
    else if (t[1]==vertexIndex)
        return 1;
    else if (t[2]==vertexIndex)
        return 2;
    else if (t[3]==vertexIndex)
        return 3;
    else
        return -1;
}

int TetrahedronSetTopologyContainer::getEdgeIndexInTetrahedron(const TetrahedronEdges &t,
        const unsigned int edgeIndex) const
{
    if (t[0]==edgeIndex)
        return 0;
    else if (t[1]==edgeIndex)
        return 1;
    else if (t[2]==edgeIndex)
        return 2;
    else if (t[3]==edgeIndex)
        return 3;
    else if (t[4]==edgeIndex)
        return 4;
    else if (t[5]==edgeIndex)
        return 5;
    else
        return -1;
}

int TetrahedronSetTopologyContainer::getTriangleIndexInTetrahedron(const TetrahedronTriangles &t,
        const unsigned int triangleIndex) const
{
    if (t[0]==triangleIndex)
        return 0;
    else if (t[1]==triangleIndex)
        return 1;
    else if (t[2]==triangleIndex)
        return 2;
    else if (t[3]==triangleIndex)
        return 3;
    else
        return -1;
}

sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronEdgeShellForModification(const unsigned int i)
{
    if (!hasTetrahedronEdgeShell())
        createTetrahedronEdgeShellArray();

    assert(i < m_tetrahedronEdgeShell.size());

    return m_tetrahedronEdgeShell[i];
}

sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronVertexShellForModification(const unsigned int i)
{
    if (!hasTetrahedronVertexShell())
        createTetrahedronVertexShellArray();

    assert(i < m_tetrahedronVertexShell.size());

    return m_tetrahedronVertexShell[i];
}

sofa::helper::vector< unsigned int > &TetrahedronSetTopologyContainer::getTetrahedronTriangleShellForModification(const unsigned int i)
{
    if (!hasTetrahedronTriangleShell())
        createTetrahedronTriangleShellArray();

    assert(i < m_tetrahedronTriangleShell.size());

    return m_tetrahedronTriangleShell[i];
}


bool TetrahedronSetTopologyContainer::checkTopology() const
{
#ifndef NDEBUG
    bool ret = true;

    if(hasTetrahedronVertexShell())
    {
        for (unsigned int i=0; i<m_tetrahedronVertexShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tvs = m_tetrahedronVertexShell[i];
            for (unsigned int j=0; j<tvs.size(); ++j)
            {
                bool check_tetra_vertex_shell = (m_tetrahedron[tvs[j]][0]==i)
                        ||  (m_tetrahedron[tvs[j]][1]==i)
                        ||  (m_tetrahedron[tvs[j]][2]==i)
                        ||  (m_tetrahedron[tvs[j]][3]==i);
                if(!check_tetra_vertex_shell)
                {
                    std::cout << "*** CHECK FAILED : check_tetra_vertex_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    if (hasTetrahedronEdgeShell())
    {
        for (unsigned int i=0; i<m_tetrahedronEdgeShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_tetrahedronEdgeShell[i];
            for (unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_tetra_edge_shell =  (m_tetrahedronEdge[tes[j]][0]==i)
                        || (m_tetrahedronEdge[tes[j]][1]==i)
                        || (m_tetrahedronEdge[tes[j]][2]==i)
                        || (m_tetrahedronEdge[tes[j]][3]==i)
                        || (m_tetrahedronEdge[tes[j]][4]==i)
                        || (m_tetrahedronEdge[tes[j]][5]==i);
                if(!check_tetra_edge_shell)
                {
                    std::cout << "*** CHECK FAILED : check_tetra_edge_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    if (hasTetrahedronTriangleShell())
    {
        for (unsigned int i=0; i<m_tetrahedronTriangleShell.size(); ++i)
        {
            const sofa::helper::vector<unsigned int> &tes=m_tetrahedronTriangleShell[i];
            for (unsigned int j=0; j<tes.size(); ++j)
            {
                bool check_tetra_triangle_shell =  (m_tetrahedronTriangle[tes[j]][0]==i)
                        || (m_tetrahedronTriangle[tes[j]][1]==i)
                        || (m_tetrahedronTriangle[tes[j]][2]==i)
                        || (m_tetrahedronTriangle[tes[j]][3]==i);
                if(!check_tetra_triangle_shell)
                {
                    std::cout << "*** CHECK FAILED : check_tetra_triangle_shell, i = " << i << " , j = " << j << std::endl;
                    ret = false;
                }
            }
        }
    }

    return ret && TriangleSetTopologyContainer::checkTopology();
#else
    return true;
#endif
}

bool TetrahedronSetTopologyContainer::hasTetrahedra() const
{
    d_tetrahedron.getValue(); // make sure m_tetrahedron is valid
    return !m_tetrahedron.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedronEdges() const
{
    return !m_tetrahedronEdge.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedronTriangles() const
{
    return !m_tetrahedronTriangle.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedronVertexShell() const
{
    return !m_tetrahedronVertexShell.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedronEdgeShell() const
{
    return !m_tetrahedronEdgeShell.empty();
}

bool TetrahedronSetTopologyContainer::hasTetrahedronTriangleShell() const
{
    return !m_tetrahedronTriangleShell.empty();
}

void TetrahedronSetTopologyContainer::clearTetrahedra()
{
    d_tetrahedron.beginEdit();
    m_tetrahedron.clear();
    d_tetrahedron.endEdit();
}

void TetrahedronSetTopologyContainer::clearTetrahedronEdges()
{
    m_tetrahedronEdge.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedronTriangles()
{
    m_tetrahedronTriangle.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedronVertexShell()
{
    m_tetrahedronVertexShell.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedronEdgeShell()
{
    m_tetrahedronEdgeShell.clear();
}

void TetrahedronSetTopologyContainer::clearTetrahedronTriangleShell()
{
    m_tetrahedronTriangleShell.clear();
}

void TetrahedronSetTopologyContainer::clear()
{
    clearTetrahedronVertexShell();
    clearTetrahedronEdgeShell();
    clearTetrahedronTriangleShell();
    clearTetrahedronEdges();
    clearTetrahedronTriangles();
    clearTetrahedra();

    TriangleSetTopologyContainer::clear();
}

} // namespace topology

} // namespace component

} // namespace sofa

