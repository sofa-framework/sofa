/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <SofaBaseTopology/MeshTopology.h>
#include <sofa/core/visual/VisualParams.h>
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

using helper::vector;


MeshTopology::EdgeUpdate::EdgeUpdate(MeshTopology* t)
    :PrimitiveUpdate(t)
{
    if( topology->hasVolume() )
    {
        addInput(&topology->seqHexahedra);
        addInput(&topology->seqTetrahedra);
        addOutput(&topology->seqEdges);
        setDirtyValue();
    }
    else if( topology->hasSurface() )
    {
        addInput(&topology->seqTriangles);
        addInput(&topology->seqQuads);
        addOutput(&topology->seqEdges);
        setDirtyValue();
    }

}

void MeshTopology::EdgeUpdate::update()
{
    if(topology->hasVolume() ) updateFromVolume();
    else if(topology->hasSurface()) updateFromSurface();
}

void MeshTopology::EdgeUpdate::updateFromVolume()
{
    typedef MeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef MeshTopology::SeqHexahedra  SeqHexahedra;
    typedef MeshTopology::SeqEdges     SeqEdges;
    typedef MeshTopology::Tetra Tetra;
    typedef MeshTopology::Hexa Hexa;

    SeqEdges& seqEdges = *topology->seqEdges.beginEdit();
    seqEdges.clear();
    std::map<Edge,unsigned int> edgeMap;
    unsigned int edgeIndex;

    const SeqTetrahedra& tetrahedra = topology->getTetrahedra(); // do not use seqTetrahedra directly as it might not be up-to-date
    const unsigned int edgesInTetrahedronArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        const Tetra &t=tetrahedra[i];
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        for (unsigned int j=0; j<6; ++j)
        {
            unsigned int v1=t[edgesInTetrahedronArray[j][0]];
            unsigned int v2=t[edgesInTetrahedronArray[j][1]];
            // sort vertices in lexicographics order
            if (v1<v2)
                e=Edge(v1,v2);
            else
                e=Edge(v2,v1);

            ite=edgeMap.find(e);
            if (ite==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                edgeIndex=(unsigned int)seqEdges.size();
                edgeMap[e]=edgeIndex;
                seqEdges.push_back(e);
            }
//            else
//            {
//                edgeIndex=(*ite).second;
//            }
            //m_edgesInTetrahedron[i][j]=edgeIndex;
        }
    }

    // fjourdes :
    // should the edgeMap be cleared here ? Sounds strange but it seems that is what was done in previous method.

    const SeqHexahedra& hexahedra = topology->getHexahedra(); // do not use seqHexahedra directly as it might not be up-to-date
    const unsigned int edgeHexahedronDescriptionArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};
    // create a temporary map to find redundant edges

    /// create the m_edge array at the same time than it fills the m_edgesInHexahedron array
    for (unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        const Hexa &h=hexahedra[i];
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        for (unsigned int j=0; j<12; ++j)
        {
            unsigned int v1=h[edgeHexahedronDescriptionArray[j][0]];
            unsigned int v2=h[edgeHexahedronDescriptionArray[j][1]];
            // sort vertices in lexicographics order
            if (v1<v2)
                e=Edge(v1,v2);
            else
                e=Edge(v2,v1);

            ite=edgeMap.find(e);
            if (ite==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                edgeIndex=(unsigned int)seqEdges.size();
                edgeMap[e]=edgeIndex;
                seqEdges.push_back(e);

            }
//            else
//            {
//                edgeIndex=(*ite).second;
//            }
            //m_edgesInHexahedron[i][j]=edgeIndex;
        }
    }
    topology->seqEdges.endEdit();
}

void MeshTopology::EdgeUpdate::updateFromSurface()
{
    typedef MeshTopology::SeqTriangles SeqTriangles;
    typedef MeshTopology::SeqQuads     SeqQuads;
    typedef MeshTopology::SeqEdges     SeqEdges;
    typedef MeshTopology::Triangle     Triangle;
    typedef MeshTopology::Quad         Quad;

    std::map<Edge,unsigned int> edgeMap;
    unsigned int edgeIndex;
    SeqEdges& seqEdges = *topology->seqEdges.beginEdit();
    seqEdges.clear();
    const SeqTriangles& triangles = topology->getTriangles(); // do not use seqTriangles directly as it might not be up-to-date
    for (unsigned int i = 0; i < triangles.size(); ++i)
    {
        const Triangle &t=triangles[i];
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        for (unsigned int j=0; j<3; ++j)
        {
            unsigned int v1=t[(j+1)%3];
            unsigned int v2=t[(j+2)%3];
            // sort vertices in lexicographics order
            if (v1<v2)
                e=Edge(v1,v2);
            else
                e=Edge(v2,v1);
            ite=edgeMap.find(e);
            if (ite==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                edgeIndex=(unsigned int)seqEdges.size();
                edgeMap[e]=edgeIndex;
                seqEdges.push_back(e);
            }
//            else
//            {
//                edgeIndex=(*ite).second;
//            }
            //m_edgesInTriangle[i][j]=edgeIndex;
        }
    }

    const SeqQuads& quads = topology->getQuads(); // do not use seqQuads directly as it might not be up-to-date
    for (unsigned int i = 0; i < quads.size(); ++i)
    {
        const Quad &t=quads[i];
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        for (unsigned int j=0; j<4; ++j)
        {
            unsigned int v1=t[(j+1)%4];
            unsigned int v2=t[(j+2)%4];
            // sort vertices in lexicographics order
            if (v1<v2)
                e=Edge(v1,v2);
            else
                e=Edge(v2,v1);
            ite=edgeMap.find(e);
            if (ite==edgeMap.end())
            {
                // edge not in edgeMap so create a new one
                edgeIndex=(unsigned int)seqEdges.size();
                edgeMap[e]=edgeIndex;
                seqEdges.push_back(e);
            }
//            else
//            {
//                edgeIndex=(*ite).second;
//            }
            //m_edgesInQuad[i][j]=edgeIndex;
        }
    }

    topology->seqEdges.endEdit();
}


MeshTopology::TriangleUpdate::TriangleUpdate(MeshTopology *t)
    :PrimitiveUpdate(t)
{
    addInput(&topology->seqTetrahedra);
    addOutput(&topology->seqTriangles);
    setDirtyValue();
}


void MeshTopology::TriangleUpdate::update()
{
    typedef MeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef MeshTopology::SeqTriangles SeqTriangles;
    const SeqTetrahedra& tetrahedra = topology->getTetrahedra();
    SeqTriangles& seqTriangles = *topology->seqTriangles.beginEdit();
    seqTriangles.clear();
    // create a temporary map to find redundant triangles
    std::map<Triangle,unsigned int> triangleMap;
    std::map<Triangle,unsigned int>::iterator itt;
    Triangle tr;
    unsigned int triangleIndex;
    unsigned int v[3],val;
    /// create the m_edge array at the same time than it fills the m_trianglesInTetrahedron array
    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        const Tetra &t=topology->seqTetrahedra.getValue()[i];
        for (unsigned int j=0; j<4; ++j)
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
            tr=Triangle(v[0],v[2],v[1]);
            itt=triangleMap.find(tr);
            if (itt==triangleMap.end())
            {
                // edge not in edgeMap so create a new one
                triangleIndex=(unsigned int)seqTriangles.size();
                tr=Triangle(v[0],v[1],v[2]);
                triangleMap[tr]=triangleIndex;
                seqTriangles.push_back(tr);
            }
//            else
//            {
//                triangleIndex=(*itt).second;
//            }
            //m_trianglesInTetrahedron[i][j]=triangleIndex;
        }
    }

    topology->seqTriangles.endEdit();

}

MeshTopology::QuadUpdate::QuadUpdate(MeshTopology *t)
    :PrimitiveUpdate(t)
{
    addInput(&topology->seqHexahedra);
    addOutput(&topology->seqQuads);
    setDirtyValue();
}

void MeshTopology::QuadUpdate::update()
{
    typedef MeshTopology::SeqHexahedra SeqHexahedra;
    typedef MeshTopology::SeqQuads SeqQuads;

    SeqQuads& seqQuads = *topology->seqQuads.beginEdit();
    seqQuads.clear();

    if (topology->getNbHexahedra()==0) return; // no hexahedra to extract edges from

    const SeqHexahedra& hexahedra = topology->getHexahedra(); // do not use seqQuads directly as it might not be up-to-date

    // create a temporary map to find redundant quads
    std::map<Quad,unsigned int> quadMap;
    std::map<Quad,unsigned int>::iterator itt;
    Quad qu;
    unsigned int v[4],val;
    unsigned int quadIndex;
    /// create the m_edge array at the same time than it fills the m_edgesInHexahedron array
    for (unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        const Hexa &h=hexahedra[i];

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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if (itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)seqQuads.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;

            seqQuads.push_back(qu);
        }
//        else
//        {
//            quadIndex=(*itt).second;
//        }
        //m_quadsInHexahedron[i][0]=quadIndex;

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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if (itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)seqQuads.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            seqQuads.push_back(qu);

        }
//        else
//        {
//            quadIndex=(*itt).second;
//        }
        //m_quadsInHexahedron[i][1]=quadIndex;

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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if (itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)seqQuads.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            seqQuads.push_back(qu);
        }
//        else
//        {
//            quadIndex=(*itt).second;
//        }
        //m_quadsInHexahedron[i][2]=quadIndex;

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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if (itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)topology->seqQuads.getValue().size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            seqQuads.push_back(qu);
        }
//        else
//        {
//            quadIndex=(*itt).second;
//        }
        //m_quadsInHexahedron[i][3]=quadIndex;

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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if (itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)seqQuads.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            seqQuads.push_back(qu);
        }
//        else
//        {
//            quadIndex=(*itt).second;
//        }
        //m_quadsInHexahedron[i][4]=quadIndex;

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
        qu=Quad(v[0],v[3],v[2],v[1]);
        itt=quadMap.find(qu);
        if (itt==quadMap.end())
        {
            // quad not in edgeMap so create a new one
            quadIndex=(unsigned int)seqQuads.size();
            quadMap[qu]=quadIndex;
            qu=Quad(v[0],v[1],v[2],v[3]);
            quadMap[qu]=quadIndex;
            seqQuads.push_back(qu);
        }
//        else
//        {
//            quadIndex=(*itt).second;
//        }
        //m_quadsInHexahedron[i][5]=quadIndex;
    }

    topology->seqQuads.endEdit();
}

using namespace sofa::defaulttype;
using core::topology::BaseMeshTopology;


SOFA_DECL_CLASS(MeshTopology)

int MeshTopologyClass = core::RegisterObject("Generic mesh topology")
        .addAlias("Mesh")
        .add< MeshTopology >()
        ;

MeshTopology::MeshTopology()
    : seqPoints(initData(&seqPoints,"position","List of point positions"))
    , seqEdges(initData(&seqEdges,"edges","List of edge indices"))
    , seqTriangles(initData(&seqTriangles,"triangles","List of triangle indices"))
    , seqQuads(initData(&seqQuads,"quads","List of quad indices"))
    , seqTetrahedra(initData(&seqTetrahedra,"tetrahedra","List of tetrahedron indices"))
    , isToPrint( initData(&isToPrint, false, "isToPrint", "suppress somes data before using save as function"))
    , seqHexahedra(initData(&seqHexahedra,"hexahedra","List of hexahedron indices"))
    , seqUVs(initData(&seqUVs,"uv","List of uv coordinates"))
    , nbPoints(0)
    , validTetrahedra(false), validHexahedra(false)
    , revision(0)
    , _drawEdges(initData(&_drawEdges, false, "drawEdges","if true, draw the topology Edges"))
    , _drawTriangles(initData(&_drawTriangles, false, "drawTriangles","if true, draw the topology Triangles"))
    , _drawQuads(initData(&_drawQuads, false, "drawQuads","if true, draw the topology Quads"))
    , _drawTetra(initData(&_drawTetra, false, "drawTetrahedra","if true, draw the topology Tetrahedra"))
    , _drawHexa(initData(&_drawHexa, false, "drawHexahedra","if true, draw the topology hexahedra"))
    , UpperTopology(sofa::core::topology::EDGE)
{
    addAlias(&seqPoints,"points");
    addAlias(&seqEdges,"lines");
    addAlias(&seqTetrahedra,"tetras");
    addAlias(&seqHexahedra,"hexas");
    addAlias(&seqUVs,"texcoords");
}

void MeshTopology::init()
{
    if(isToPrint.getValue()==true) seqEdges.setPersistent(false);
    BaseMeshTopology::init();
    if (nbPoints==0)
    {
        // looking for upper topology
        if (!seqHexahedra.getValue().empty())
            UpperTopology = sofa::core::topology::HEXAHEDRON;
        else if (!seqTetrahedra.getValue().empty())
            UpperTopology = sofa::core::topology::TETRAHEDRON;
        else if (!seqQuads.getValue().empty())
            UpperTopology = sofa::core::topology::QUAD;
        else if (!seqTriangles.getValue().empty())
            UpperTopology = sofa::core::topology::TRIANGLE;
        else
            UpperTopology = sofa::core::topology::EDGE;
    }

    // compute the number of points, if the topology is charged from the scene or if it was loaded from a MeshLoader without any points data.
    if (nbPoints==0)
    {
        unsigned int n = 0;
        for (unsigned int i=0; i<seqEdges.getValue().size(); i++)
        {
            for (unsigned int j=0; j<seqEdges.getValue()[i].size(); j++)
            {
                if (n <= seqEdges.getValue()[i][j])
                    n = 1 + seqEdges.getValue()[i][j];
            }
        }
        for (unsigned int i=0; i<seqTriangles.getValue().size(); i++)
        {
            for (unsigned int j=0; j<seqTriangles.getValue()[i].size(); j++)
            {
                if (n <= seqTriangles.getValue()[i][j])
                    n = 1 + seqTriangles.getValue()[i][j];
            }
        }
        for (unsigned int i=0; i<seqQuads.getValue().size(); i++)
        {
            for (unsigned int j=0; j<seqQuads.getValue()[i].size(); j++)
            {
                if (n <= seqQuads.getValue()[i][j])
                    n = 1 + seqQuads.getValue()[i][j];
            }
        }
        for (unsigned int i=0; i<seqTetrahedra.getValue().size(); i++)
        {
            for (unsigned int j=0; j<seqTetrahedra.getValue()[i].size(); j++)
            {
                if (n <= seqTetrahedra.getValue()[i][j])
                    n = 1 + seqTetrahedra.getValue()[i][j];
            }
        }
        for (unsigned int i=0; i<seqHexahedra.getValue().size(); i++)
        {
            for (unsigned int j=0; j<seqHexahedra.getValue()[i].size(); j++)
            {
                if (n <= seqHexahedra.getValue()[i][j])
                    n = 1 + seqHexahedra.getValue()[i][j];
            }
        }

        nbPoints = n;
    }

    if(seqEdges.getValue().empty() )
    {
        if(seqEdges.getParent() != NULL )
        {
            seqEdges.delInput(seqEdges.getParent());
        }
        EdgeUpdate::SPtr edgeUpdate = sofa::core::objectmodel::New<EdgeUpdate>(this);
        edgeUpdate->setName("edgeUpdate");
        this->addSlave(edgeUpdate);
    }
    if(seqTriangles.getValue().empty() )
    {
        if(seqTriangles.getParent() != NULL)
        {
            seqTriangles.delInput(seqTriangles.getParent());
        }
        TriangleUpdate::SPtr triangleUpdate = sofa::core::objectmodel::New<TriangleUpdate>(this);
        triangleUpdate->setName("triangleUpdate");
        this->addSlave(triangleUpdate);
    }
    if(seqQuads.getValue().empty() )
    {
        if(seqQuads.getParent() != NULL )
        {
            seqQuads.delInput(seqQuads.getParent());
        }
        QuadUpdate::SPtr quadUpdate = sofa::core::objectmodel::New<QuadUpdate>(this);
        quadUpdate->setName("quadUpdate");
        this->addSlave(quadUpdate);
    }
}

void MeshTopology::clear()
{
    nbPoints = 0;
    seqPoints.beginWriteOnly()->clear(); seqPoints.endEdit();
    seqEdges.beginWriteOnly()->clear(); seqEdges.endEdit();
    seqTriangles.beginWriteOnly()->clear(); seqTriangles.endEdit();
    seqQuads.beginWriteOnly()->clear(); seqQuads.endEdit();
    seqTetrahedra.beginWriteOnly()->clear(); seqTetrahedra.endEdit();
    seqHexahedra.beginWriteOnly()->clear(); seqHexahedra.endEdit();

    seqUVs.beginWriteOnly()->clear(); seqUVs.endEdit();

    invalidate();
}


void MeshTopology::addPoint(SReal px, SReal py, SReal pz)
{
    seqPoints.beginEdit()->push_back(defaulttype::Vec<3,SReal>((SReal)px, (SReal)py, (SReal)pz));
    seqPoints.endEdit();
    if (seqPoints.getValue().size() > (size_t)nbPoints)
        nbPoints = (int)seqPoints.getValue().size();
}

void MeshTopology::addEdge( int a, int b )
{
    seqEdges.beginEdit()->push_back(Edge(a,b));
    seqEdges.endEdit();
    if (a >= (int)nbPoints) nbPoints = a+1;
    if (b >= (int)nbPoints) nbPoints = b+1;
}

void MeshTopology::addTriangle( int a, int b, int c )
{
    seqTriangles.beginEdit()->push_back( Triangle(a,b,c) );
    seqTriangles.endEdit();
    if (a >= (int)nbPoints) nbPoints = a+1;
    if (b >= (int)nbPoints) nbPoints = b+1;
    if (c >= (int)nbPoints) nbPoints = c+1;
}

void MeshTopology::addQuad(int a, int b, int c, int d)
{
    seqQuads.beginEdit()->push_back(Quad(a,b,c,d));
    seqQuads.endEdit();
    if (a >= (int)nbPoints) nbPoints = a+1;
    if (b >= (int)nbPoints) nbPoints = b+1;
    if (c >= (int)nbPoints) nbPoints = c+1;
    if (d >= (int)nbPoints) nbPoints = d+1;
}

void MeshTopology::addTetra( int a, int b, int c, int d )
{
    seqTetrahedra.beginEdit()->push_back( Tetra(a,b,c,d) );
    seqTetrahedra.endEdit();
    if (a >= (int)nbPoints) nbPoints = a+1;
    if (b >= (int)nbPoints) nbPoints = b+1;
    if (c >= (int)nbPoints) nbPoints = c+1;
    if (d >= (int)nbPoints) nbPoints = d+1;
}

void MeshTopology::addHexa(int p1, int p2, int p3, int p4, int p5, int p6, int p7, int p8)
{
#ifdef SOFA_NEW_HEXA
    seqHexahedra.beginEdit()->push_back(Hexa(p1,p2,p3,p4,p5,p6,p7,p8));
#else
    seqHexahedra.beginEdit()->push_back(Hexa(p1,p2,p4,p3,p5,p6,p8,p7));
#endif
    seqHexahedra.endEdit();
    if (p1 >= (int)nbPoints) nbPoints = p1+1;
    if (p2 >= (int)nbPoints) nbPoints = p2+1;
    if (p3 >= (int)nbPoints) nbPoints = p3+1;
    if (p4 >= (int)nbPoints) nbPoints = p4+1;
    if (p5 >= (int)nbPoints) nbPoints = p5+1;
    if (p6 >= (int)nbPoints) nbPoints = p6+1;
    if (p7 >= (int)nbPoints) nbPoints = p7+1;
    if (p8 >= (int)nbPoints) nbPoints = p8+1;
}

void MeshTopology::addUV(SReal u, SReal v)
{
    seqUVs.beginEdit()->push_back(defaulttype::Vec<2,SReal>((SReal)u, (SReal)v));
    seqUVs.endEdit();
    if (seqUVs.getValue().size() > (size_t)nbPoints)
        nbPoints = (int)seqUVs.getValue().size();
}

const MeshTopology::SeqEdges& MeshTopology::getEdges()
{
    return seqEdges.getValue();
}

const MeshTopology::SeqTriangles& MeshTopology::getTriangles()
{
    return seqTriangles.getValue();
}

const MeshTopology::SeqQuads& MeshTopology::getQuads()
{

    return seqQuads.getValue();
}

const MeshTopology::SeqTetrahedra& MeshTopology::getTetrahedra()
{
    if (!validTetrahedra)
    {
        updateTetrahedra();
        validTetrahedra = true;
    }
    return seqTetrahedra.getValue();
}

const MeshTopology::SeqHexahedra& MeshTopology::getHexahedra()
{
    if (!validHexahedra)
    {
        updateHexahedra();
        validHexahedra = true;
    }
    return seqHexahedra.getValue();
}

const MeshTopology::SeqUV& MeshTopology::getUVs()
{
    return seqUVs.getValue();
}

int MeshTopology::getNbPoints() const
{
    return nbPoints;
}

void MeshTopology::setNbPoints(int n)
{
    nbPoints = n;
}

int MeshTopology::getNbEdges()
{
    return (int)getEdges().size();
}

int MeshTopology::getNbTriangles()
{
    return (int)getTriangles().size();
}

int MeshTopology::getNbQuads()
{
    return (int)getQuads().size();
}

int MeshTopology::getNbTetrahedra()
{
    return (int)getTetrahedra().size();
}

int MeshTopology::getNbHexahedra()
{
    return (int)getHexahedra().size();
}

int MeshTopology::getNbUVs()
{
    return (int)getUVs().size();
}

const MeshTopology::Edge MeshTopology::getEdge(index_type i)
{
    return getEdges()[i];
}

const MeshTopology::Triangle MeshTopology::getTriangle(index_type i)
{
    return getTriangles()[i];
}

const MeshTopology::Quad MeshTopology::getQuad(index_type i)
{
    return getQuads()[i];
}

const MeshTopology::Tetra MeshTopology::getTetrahedron(index_type i)
{
    return getTetrahedra()[i];
}

const MeshTopology::Hexa MeshTopology::getHexahedron(index_type i)
{
    return getHexahedra()[i];
}

const MeshTopology::UV MeshTopology::getUV(index_type i)
{
    return getUVs()[i];
}

void MeshTopology::createEdgesAroundVertexArray ()
{
    const SeqEdges& edges = getEdges();
    m_edgesAroundVertex.clear();
    m_edgesAroundVertex.resize( nbPoints );
    /*....
        if (getNbTetrahedra() || getNbHexahedra())
        { // Unordered shells if the mesh is volumic
            for (unsigned int i = 0; i < edges.size(); ++i)
            {
                // adding edge i in the edge shell of both points
                m_edgesAroundVertex[ edges[i][0] ].push_back( i );
                m_edgesAroundVertex[ edges[i][1] ].push_back( i );
            }
        }
        else if (getNbTriangles() || getNbQuads())
        { // order edges in consistent order if possible (i.e. on manifold meshes)
            bool createdTriangleShell = getNbTriangles() && m_edgesInTriangle.empty();
            bool createdQuadShell = getNbQuads() && m_edgesInQuad.empty();
            if (createdTriangleShell) createTrianglesAroundVertexArray();
            if (createdQuadShell) createQuadsAroundVertexArray();
            const SeqTriangles& triangles = getTriangles();
            const SeqQuads& quads = getQuads();

        }
        else*/
    {
        // 1D mesh : put inbound edges before outbound edges
        for (unsigned int i = 0; i < edges.size(); ++i)
        {
            // adding edge i in the edge shell of both points
            m_edgesAroundVertex[ edges[i][0] ].push_back( i );
            m_edgesAroundVertex[ edges[i][1] ].insert( m_edgesAroundVertex[ edges[i][1] ].begin(), i );
        }
    }
}

void MeshTopology::createEdgesInTriangleArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use seqEdges directly as it might not be up-to-date
    const SeqTriangles& triangles = getTriangles(); // do not use seqTriangles directly as it might not be up-to-date
    m_edgesInTriangle.clear();
    m_edgesInTriangle.resize(triangles.size());
    for (unsigned int i = 0; i < triangles.size(); ++i)
    {
        const Triangle &t=triangles[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<3; ++j)
        {
            int edgeIndex=getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
            assert(edgeIndex!= -1);
            m_edgesInTriangle[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createEdgesInQuadArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use seqEdges directly as it might not be up-to-date
    const SeqQuads& quads = getQuads(); // do not use seqQuads directly as it might not be up-to-date
    m_edgesInQuad.clear();
    m_edgesInQuad.resize(quads.size());
    for (unsigned int i = 0; i < quads.size(); ++i)
    {
        const Quad &t=quads[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<4; ++j)
        {
            int edgeIndex=getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
            assert(edgeIndex!= -1);
            m_edgesInQuad[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createEdgesInTetrahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use seqEdges directly as it might not be up-to-date
    const SeqTetrahedra& tetrahedra = getTetrahedra(); // do not use seqTetrahedra directly as it might not be up-to-date
    m_edgesInTetrahedron.clear();
    m_edgesInTetrahedron.resize(tetrahedra.size());
    const unsigned int edgesInTetrahedronArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};

    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        const Tetra &t=tetrahedra[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<6; ++j)
        {
            int edgeIndex=getEdgeIndex(t[edgesInTetrahedronArray[j][0]],
                    t[edgesInTetrahedronArray[j][1]]);
            assert(edgeIndex!= -1);
            m_edgesInTetrahedron[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createEdgesInHexahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use seqEdges directly as it might not be up-to-date
    //getEdges();
    const SeqHexahedra& hexahedra = getHexahedra(); // do not use seqHexahedra directly as it might not be up-to-date
    m_edgesInHexahedron.clear();
    m_edgesInHexahedron.resize(hexahedra.size());
    const unsigned int edgeHexahedronDescriptionArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};

    for (unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        const Hexa &h=hexahedra[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<12; ++j)
        {
            int edgeIndex=getEdgeIndex(h[edgeHexahedronDescriptionArray[j][0]],
                    h[edgeHexahedronDescriptionArray[j][1]]);
            assert(edgeIndex!= -1);
            m_edgesInHexahedron[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createTrianglesAroundVertexArray ()
{
    const SeqTriangles& triangles = getTriangles(); // do not use seqTriangles directly as it might not be up-to-date
    m_trianglesAroundVertex.clear();
    m_trianglesAroundVertex.resize( nbPoints );

    for (unsigned int i = 0; i < triangles.size(); ++i)
    {
        // adding triangle i in the triangle shell of all points
        for (unsigned int j=0; j<3; ++j)
            m_trianglesAroundVertex[ triangles[i][j]  ].push_back( i );
    }
}

void MeshTopology::createOrientedTrianglesAroundVertexArray()
{
    if (m_edgesAroundVertex.size() == 0)
        createEdgesAroundVertexArray();

    const SeqEdges& edges = getEdges();
    m_orientedTrianglesAroundVertex.clear();
    m_orientedTrianglesAroundVertex.resize(nbPoints);
    m_orientedEdgesAroundVertex.clear();
    m_orientedEdgesAroundVertex.resize(nbPoints);

    for(unsigned int i = 0; i < (unsigned int)nbPoints; ++i)
        //for each point: i
    {
        unsigned int startEdge = InvalidID;
        unsigned int currentEdge = InvalidID;
        unsigned int nextEdge = InvalidID;
        unsigned int lastTri = InvalidID;

        // skip points not attached to any edge
        if (m_edgesAroundVertex[i].empty()) continue;

        //find the start edge for a boundary point
        for(unsigned int j = 0; j < m_edgesAroundVertex[i].size() && startEdge == InvalidID; ++j)
            //for each edge adjacent to the point: m_edgesAroundVertex[i][j]
        {
            const TrianglesAroundEdge& eTris = getTrianglesAroundEdge(m_edgesAroundVertex[i][j]);
            if(eTris.size() == 1)
                //m_edgesAroundVertex[i][j] is a boundary edge, test whether it is the start edge
            {
                //find out if there is a next edge in the right orientation around the point i
                const EdgesInTriangle& tEdges = getEdgesInTriangle(eTris[0]);
                for(unsigned int k = 0; k < tEdges.size() && startEdge == InvalidID; ++k)
                    //for each edge of the triangle: tEdges[k]
                {
                    if(tEdges[k] != m_edgesAroundVertex[i][j])
                        // pick up the edge which is not the current one
                    {
                        for(unsigned int p = 0; p < 2; ++p)
                            //for each end point of the edge: edges[tEdges[k]][p]
                        {
                            if(edges[tEdges[k]][p] == i)
                                // pick up the edge starting from point i
                            {
                                if(-1 == computeRelativeOrientationInTri(i, edges[tEdges[k]][(p+1)%2], eTris[0]))
                                    // pick up the edge with the consistent orientation (the same orientation as the triangle)
                                {
                                    startEdge = m_edgesAroundVertex[i][j];
                                    currentEdge = startEdge;
                                    nextEdge = tEdges[k];
                                    m_orientedTrianglesAroundVertex[i].push_back(eTris[0]);
                                    m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                                    lastTri = eTris[0];
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        //set a start edge for the non-boundary point
        if(startEdge == InvalidID)
        {
            startEdge = m_edgesAroundVertex[i][0];
            currentEdge = startEdge;
            //find the next edge around the point i
            const TrianglesAroundEdge& eTris = getTrianglesAroundEdge(currentEdge);
            for(unsigned int j = 0; j < eTris.size() && nextEdge == InvalidID; ++j)
                //for each triangle adjacent to the currentEdge: eTris[j]
            {
                const EdgesInTriangle& tEdges = getEdgesInTriangle(eTris[j]);
                for(unsigned int k = 0; k < tEdges.size() && nextEdge == InvalidID; ++k)
                    //for each edge of the triangle: tEdges[k]
                {
                    if(tEdges[k] != currentEdge)
                        // pick up the edge which is not the current one
                    {
                        for(unsigned int p = 0; p < 2; ++p)
                            //for each end point of the edge: edges[tEdges[k]][p]
                        {
                            if(edges[tEdges[k]][p] == i)
                                // pick up the edge starting from point i
                            {
                                if(-1 == computeRelativeOrientationInTri(i, edges[tEdges[k]][(p+1)%2], eTris[j]))
                                    // pick up the edge with the consistent orientation (the same orientation as the triangle)
                                {
                                    nextEdge = tEdges[k];
                                    m_orientedTrianglesAroundVertex[i].push_back(eTris[j]);
                                    m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                                    lastTri = eTris[j];
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        //begin the loop to find the next edge around the point i
        currentEdge = nextEdge;
        nextEdge = InvalidID;
        while(currentEdge != startEdge && currentEdge != InvalidID)
        {
            const TrianglesAroundEdge& eTris = getTrianglesAroundEdge(currentEdge);
            if(eTris.size() == 1)
            {
                m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                break;
            }
            for(unsigned int j = 0; j < eTris.size() && nextEdge == InvalidID; ++j)
                // for each triangle adjacent to the currentEdge: eTris[j]
            {
                if(eTris[j] != lastTri)
                {
                    m_orientedTrianglesAroundVertex[i].push_back(eTris[j]);
                    m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                    lastTri = eTris[j];
                    //find the nextEdge
                    const EdgesInTriangle& tEdges = getEdgesInTriangle(eTris[j]);
                    for(unsigned int k = 0; k < tEdges.size(); ++k)
                    {
                        if(tEdges[k] != currentEdge && (edges[tEdges[k]][0] == i || edges[tEdges[k]][1] == i))
                        {
                            nextEdge = tEdges[k];
                            break;
                        }
                    }
                }
            }
            currentEdge = nextEdge;
            nextEdge = InvalidID;
            // FIX: check is currentEdge is not already in orientedEdgesAroundVertex to avoid infinite loops in case of non manifold topology
            for (unsigned int j = 0; i < m_orientedEdgesAroundVertex[i].size(); ++i)
            {
                if (m_orientedEdgesAroundVertex[i][j] == currentEdge)
                {
                    currentEdge = InvalidID; // go out of the while loop
                    break;
        }
    }
}
    }
}

void MeshTopology::createTrianglesAroundEdgeArray ()
{
    const SeqTriangles& triangles = getTriangles(); // do not use seqTriangles directly as it might not be up-to-date
    if (m_edgesInTriangle.empty())
        createEdgesInTriangleArray();
    m_trianglesAroundEdge.clear();
    m_trianglesAroundEdge.resize( getNbEdges());
    const vector< EdgesInTriangle > &tea=m_edgesInTriangle;
    unsigned int j;

    for (unsigned int i = 0; i < triangles.size(); ++i)
    {
        const Triangle &t=triangles[i];
        // adding triangle i in the triangle shell of all edges
        for (j=0; j<3; ++j)
        {
            if (seqEdges.getValue()[tea[i][j]][0] == t[(j+1)%3])
                m_trianglesAroundEdge[ tea[i][j] ].insert(m_trianglesAroundEdge[ tea[i][j] ].begin(), i); // triangle is on the left of the edge
            else
                m_trianglesAroundEdge[ tea[i][j] ].push_back( i ); // triangle is on the right of the edge
        }
    }
}

void MeshTopology::createTrianglesInTetrahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use seqEdges directly as it might not be up-to-date
    const SeqTetrahedra& tetrahedra = getTetrahedra(); // do not use seqTetrahedra directly as it might not be up-to-date
    m_trianglesInTetrahedron.clear();
    m_trianglesInTetrahedron.resize(tetrahedra.size());

    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        const Tetra &t=tetrahedra[i];
        // adding triangles in the triangle list of the ith tetrahedron  i
        for (unsigned int j=0; j<4; ++j)
        {
            int triangleIndex=getTriangleIndex(t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
            assert(triangleIndex!= -1);
            m_trianglesInTetrahedron[i][j]=triangleIndex;
        }
    }
}


void MeshTopology::createQuadsAroundVertexArray ()
{
    const SeqQuads& quads = getQuads(); // do not use seqQuads directly as it might not be up-to-date
    m_quadsAroundVertex.clear();
    m_quadsAroundVertex.resize( nbPoints );

    for (unsigned int i = 0; i < quads.size(); ++i)
    {
        // adding quad i in the quad shell of all points
        for (unsigned j=0; j<4; ++j)
            m_quadsAroundVertex[ quads[i][j]  ].push_back( i );
    }
}

void MeshTopology::createOrientedQuadsAroundVertexArray()
{
    if(m_edgesAroundVertex.size() == 0)
        createEdgesAroundVertexArray();
    //test
    if(m_quadsAroundVertex.size() == 0)
        createQuadsAroundVertexArray();

    const SeqEdges& edges = getEdges();
    m_orientedQuadsAroundVertex.clear();
    m_orientedQuadsAroundVertex.resize(nbPoints);
    m_orientedEdgesAroundVertex.clear();
    m_orientedEdgesAroundVertex.resize(nbPoints);

    for(unsigned int i = 0; i < (unsigned int)nbPoints; ++i)
        //for each point: i
    {
        unsigned int startEdge = InvalidID;
        unsigned int currentEdge = InvalidID;
        unsigned int nextEdge = InvalidID;
        unsigned int lastQuad = InvalidID;

        //find the start edge for a boundary point
        for(unsigned int j = 0; j < m_edgesAroundVertex[i].size() && startEdge == InvalidID; ++j)
            //for each edge adjacent to the point: m_edgesAroundVertex[i][j]
        {
            const QuadsAroundEdge& eQuads = getQuadsAroundEdge(m_edgesAroundVertex[i][j]);
            if(eQuads.size() == 1)
                //m_edgesAroundVertex[i][j] is a boundary edge, test whether it is the start edge
            {
                //find out if there is a next edge in the right orientation around the point i
                const EdgesInQuad& qEdges = getEdgesInQuad(eQuads[0]);
                for(unsigned int k = 0; k < qEdges.size() && startEdge == InvalidID; ++k)
                    //for each edge of the quad: qEdges[k]
                {
                    if(qEdges[k] != m_edgesAroundVertex[i][j])
                        // pick up the edge which is not the current one
                    {
                        for(unsigned int p = 0; p < 2; ++p)
                            //for each end point of the edge: edges[qEdges[k]][p]
                        {
                            if(edges[qEdges[k]][p] == i)
                                // pick up the edge starting from point i
                            {
                                if(-1 == computeRelativeOrientationInQuad(i, edges[qEdges[k]][(p+1)%2], eQuads[0]))
                                    // pick up the edge with the consistent orientation (the same orientation as the quad)
                                {
                                    startEdge = m_edgesAroundVertex[i][j];
                                    currentEdge = startEdge;
                                    nextEdge = qEdges[k];
                                    m_orientedQuadsAroundVertex[i].push_back(eQuads[0]);
                                    m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                                    lastQuad = eQuads[0];
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        //set a start edge for the non-boundary point
        if(startEdge == InvalidID)
        {
            startEdge = m_edgesAroundVertex[i][0];
            currentEdge = startEdge;
            //find the next edge around the point i
            const QuadsAroundEdge& eQuads = getQuadsAroundEdge(currentEdge);
            for(unsigned int j = 0; j < eQuads.size() && nextEdge == InvalidID; ++j)
                //for each quad adjacent to the currentEdge: eQuads[j]
            {
                const EdgesInQuad& qEdges = getEdgesInQuad(eQuads[j]);
                for(unsigned int k = 0; k < qEdges.size() && nextEdge == InvalidID; ++k)
                    //for each edge of the quad: qEdges[k]
                {
                    if(qEdges[k] != currentEdge)
                        // pick up the edge which is not the current one
                    {
                        for(unsigned int p = 0; p < 2; ++p)
                            //for each end point of the edge: edges[qEdges[k]][p]
                        {
                            if(edges[qEdges[k]][p] == i)
                                // pick up the edge starting from point i
                            {
                                if(-1 == computeRelativeOrientationInQuad(i, edges[qEdges[k]][(p+1)%2], eQuads[j]))
                                    // pick up the edge with the consistent orientation (the same orientation as the quad)
                                {
                                    nextEdge = qEdges[k];
                                    m_orientedQuadsAroundVertex[i].push_back(eQuads[j]);
                                    m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                                    lastQuad = eQuads[j];
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        //begin the loop to find the next edge around the point i
        currentEdge = nextEdge;
        nextEdge = InvalidID;
        while(currentEdge != startEdge)
        {
            const QuadsAroundEdge& eQuads = getQuadsAroundEdge(currentEdge);
            if(eQuads.size() == 1)
            {
                m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                break;
            }
            for(unsigned int j = 0; j < eQuads.size() && nextEdge == InvalidID; ++j)
                // for each quad adjacent to the currentEdge: eQuads[j]
            {
                if(eQuads[j] != lastQuad)
                {
                    m_orientedQuadsAroundVertex[i].push_back(eQuads[j]);
                    m_orientedEdgesAroundVertex[i].push_back(currentEdge);
                    lastQuad = eQuads[j];
                    //find the nextEdge
                    const EdgesInQuad& qEdges = getEdgesInQuad(eQuads[j]);
                    for(unsigned int k = 0; k < qEdges.size(); ++k)
                    {
                        if(qEdges[k] != currentEdge && (edges[qEdges[k]][0] == i || edges[qEdges[k]][1] == i))
                        {
                            nextEdge = qEdges[k];
                            break;
                        }
                    }
                }
            }
            currentEdge = nextEdge;
            nextEdge = InvalidID;
        }
    }
}

void MeshTopology::createQuadsAroundEdgeArray ()
{
    const SeqQuads& quads = getQuads(); // do not use seqQuads directly as it might not be up-to-date
    if (m_edgesInQuad.empty())
        createEdgesInQuadArray();
    m_quadsAroundEdge.clear();
    m_quadsAroundEdge.resize( getNbEdges() );
    unsigned int j;
    for (unsigned int i = 0; i < quads.size(); ++i)
    {
        for (j=0; j<4; ++j)
            m_quadsAroundEdge[ m_edgesInQuad[i][j] ].push_back( i );
    }
}

void MeshTopology::createQuadsInHexahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use seqEdges directly as it might not be up-to-date
    const SeqHexahedra& hexahedra = getHexahedra(); // do not use seqHexahedra directly as it might not be up-to-date
    m_quadsInHexahedron.clear();
    m_quadsInHexahedron.resize(hexahedra.size());

    for (unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        const Hexa &h=hexahedra[i];
        int quadIndex;
        // adding the 6 quads in the quad list of the ith hexahedron  i
        // Quad 0 :
        quadIndex=getQuadIndex(h[0],h[3],h[2],h[1]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][0]=quadIndex;
        // Quad 1 :
        quadIndex=getQuadIndex(h[4],h[5],h[6],h[7]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][1]=quadIndex;
        // Quad 2 :
        quadIndex=getQuadIndex(h[0],h[1],h[5],h[4]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][2]=quadIndex;
        // Quad 3 :
        quadIndex=getQuadIndex(h[1],h[2],h[6],h[5]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][3]=quadIndex;
        // Quad 4 :
        quadIndex=getQuadIndex(h[2],h[3],h[7],h[6]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][4]=quadIndex;
        // Quad 5 :
        quadIndex=getQuadIndex(h[3],h[0],h[4],h[7]);
        assert(quadIndex!= -1);
        m_quadsInHexahedron[i][5]=quadIndex;
    }
}

void MeshTopology::createTetrahedraAroundVertexArray ()
{
    m_tetrahedraAroundVertex.resize( nbPoints );
    unsigned int j;

    for (unsigned int i = 0; i < seqTetrahedra.getValue().size(); ++i)
    {
        for (j=0; j<4; ++j)
            m_tetrahedraAroundVertex[ seqTetrahedra.getValue()[i][j]  ].push_back( i );
    }
}

void MeshTopology::createTetrahedraAroundEdgeArray ()
{
    if (!m_edgesInTetrahedron.size())
        createEdgesInTetrahedronArray();
    m_tetrahedraAroundEdge.resize( getNbEdges() );
    const vector< EdgesInTetrahedron > &tea = m_edgesInTetrahedron;
    unsigned int j;

    for (unsigned int i = 0; i < seqTetrahedra.getValue().size(); ++i)
    {
        for (j=0; j<6; ++j)
            m_tetrahedraAroundEdge[ tea[i][j] ].push_back( i );
    }
}

void MeshTopology::createTetrahedraAroundTriangleArray ()
{
    if (!m_trianglesInTetrahedron.size())
        createTrianglesInTetrahedronArray();
    m_tetrahedraAroundTriangle.resize( getNbTriangles());
    unsigned int j;
    const vector< TrianglesInTetrahedron > &tta=m_trianglesInTetrahedron;

    for (unsigned int i = 0; i < seqTetrahedra.getValue().size(); ++i)
    {
        for (j=0; j<4; ++j)
            m_tetrahedraAroundTriangle[ tta[i][j] ].push_back( i );
    }
}

void MeshTopology::createHexahedraAroundVertexArray ()
{
    m_hexahedraAroundVertex.resize( nbPoints );
    unsigned int j;

    for (unsigned int i = 0; i < seqHexahedra.getValue().size(); ++i)
    {
        for (j=0; j<8; ++j)
            m_hexahedraAroundVertex[ seqHexahedra.getValue()[i][j]  ].push_back( i );
    }
}

void MeshTopology::createHexahedraAroundEdgeArray ()
{
    if (!m_edgesInHexahedron.size())
        createEdgesInHexahedronArray();
    m_hexahedraAroundEdge.resize(getNbEdges());
    unsigned int j;
    const vector< EdgesInHexahedron > &hea=m_edgesInHexahedron;

    for (unsigned int i = 0; i < seqHexahedra.getValue().size(); ++i)
    {
        for (j=0; j<12; ++j)
            m_hexahedraAroundEdge[ hea[i][j] ].push_back( i );
    }
}

void MeshTopology::createHexahedraAroundQuadArray ()
{
    if (!m_quadsInHexahedron.size())
        createQuadsInHexahedronArray();
    m_hexahedraAroundQuad.resize( getNbQuads());
    unsigned int j;
    const vector< QuadsInHexahedron > &qha=m_quadsInHexahedron;

    for (unsigned int i = 0; i < seqHexahedra.getValue().size(); ++i)
    {
        // adding quad i in the edge shell of both points
        for (j=0; j<6; ++j)
            m_hexahedraAroundQuad[ qha[i][j] ].push_back( i );
    }
}

const MeshTopology::EdgesAroundVertex& MeshTopology::getEdgesAroundVertex(PointID i)
{
    if (!m_edgesAroundVertex.size() || i > m_edgesAroundVertex.size()-1)
        createEdgesAroundVertexArray();
    return m_edgesAroundVertex[i];
}

const MeshTopology::EdgesAroundVertex& MeshTopology::getOrientedEdgesAroundVertex(PointID i)
{
    if (!m_orientedEdgesAroundVertex.size() || i > m_orientedEdgesAroundVertex.size()-1)
    {
        if(getNbTriangles() != 0)
        {
            createOrientedTrianglesAroundVertexArray();
        }
        else
        {
            if(getNbQuads() != 0)
                createOrientedQuadsAroundVertexArray();
        }
    }
    return m_orientedEdgesAroundVertex[i];
}


const MeshTopology::EdgesInTriangle& MeshTopology::getEdgesInTriangle(TriangleID i)
{
    if (m_edgesInTriangle.empty() || i > m_edgesInTriangle.size()-1)
        createEdgesInTriangleArray();
    return m_edgesInTriangle[i];
}

const MeshTopology::EdgesInQuad& MeshTopology::getEdgesInQuad(QuadID i)
{
    if (m_edgesInQuad.empty() || i > m_edgesInQuad.size()-1)
        createEdgesInQuadArray();
    return m_edgesInQuad[i];
}

const MeshTopology::EdgesInTetrahedron& MeshTopology::getEdgesInTetrahedron(TetraID i)
{
    if (m_edgesInTetrahedron.empty() || i > m_edgesInTetrahedron.size()-1)
        createEdgesInTetrahedronArray();
    return m_edgesInTetrahedron[i];
}

const MeshTopology::EdgesInHexahedron& MeshTopology::getEdgesInHexahedron(HexaID i)
{
    if (!m_edgesInHexahedron.size() || i > m_edgesInHexahedron.size()-1)
        createEdgesInHexahedronArray();
    return m_edgesInHexahedron[i];
}

const MeshTopology::TrianglesAroundVertex& MeshTopology::getTrianglesAroundVertex(PointID i)
{
    if (!m_trianglesAroundVertex.size() || i > m_trianglesAroundVertex.size()-1)
        createTrianglesAroundVertexArray();
    return m_trianglesAroundVertex[i];
}
const MeshTopology::TrianglesAroundVertex& MeshTopology::getOrientedTrianglesAroundVertex(PointID i)
{
    if (!m_orientedTrianglesAroundVertex.size() || i > m_orientedTrianglesAroundVertex.size()-1)
        createOrientedTrianglesAroundVertexArray();
    return m_orientedTrianglesAroundVertex[i];
}

const MeshTopology::TrianglesAroundEdge& MeshTopology::getTrianglesAroundEdge(EdgeID i)
{
    if (m_trianglesAroundEdge.empty() || i > m_trianglesAroundEdge.size()-1)
        createTrianglesAroundEdgeArray();
    return m_trianglesAroundEdge[i];
}
const MeshTopology::TrianglesInTetrahedron& MeshTopology::getTrianglesInTetrahedron(TetraID i)
{
    if (!m_trianglesInTetrahedron.size() || i > m_trianglesInTetrahedron.size()-1)
        createTrianglesInTetrahedronArray();
    return m_trianglesInTetrahedron[i];
}

const MeshTopology::QuadsAroundVertex& MeshTopology::getQuadsAroundVertex(PointID i)
{
    if (m_quadsAroundVertex.empty() || i > m_quadsAroundVertex.size()-1)
        createQuadsAroundVertexArray();
    return m_quadsAroundVertex[i];
}

const MeshTopology::QuadsAroundVertex& MeshTopology::getOrientedQuadsAroundVertex(PointID i)
{
    if (m_orientedQuadsAroundVertex.empty() || i > m_orientedQuadsAroundVertex.size()-1)
        createOrientedQuadsAroundVertexArray();
    return m_orientedQuadsAroundVertex[i];
}

const vector< MeshTopology::QuadID >& MeshTopology::getQuadsAroundEdge(EdgeID i)
{
    if (!m_quadsAroundEdge.size() || i > m_quadsAroundEdge.size()-1)
        createQuadsAroundEdgeArray();
    return m_quadsAroundEdge[i];
}

const MeshTopology::QuadsInHexahedron& MeshTopology::getQuadsInHexahedron(HexaID i)
{
    if (!m_quadsInHexahedron.size() || i > m_quadsInHexahedron.size()-1)
        createQuadsInHexahedronArray();
    return m_quadsInHexahedron[i];
}

const MeshTopology::TetrahedraAroundVertex& MeshTopology::getTetrahedraAroundVertex(PointID i)
{
    if (!m_tetrahedraAroundVertex.size() || i > m_tetrahedraAroundVertex.size()-1)
        createTetrahedraAroundVertexArray();
    return m_tetrahedraAroundVertex[i];
}

const MeshTopology::TetrahedraAroundEdge& MeshTopology::getTetrahedraAroundEdge(EdgeID i)
{
    if (!m_tetrahedraAroundEdge.size() || i > m_tetrahedraAroundEdge.size()-1)
        createTetrahedraAroundEdgeArray();
    return m_tetrahedraAroundEdge[i];
}

const MeshTopology::TetrahedraAroundTriangle& MeshTopology::getTetrahedraAroundTriangle(TriangleID i)
{
    if (!m_tetrahedraAroundTriangle.size() || i > m_tetrahedraAroundTriangle.size()-1)
        createTetrahedraAroundTriangleArray();
    return m_tetrahedraAroundTriangle[i];
}

const MeshTopology::HexahedraAroundVertex& MeshTopology::getHexahedraAroundVertex(PointID i)
{
    if (!m_hexahedraAroundVertex.size() || i > m_hexahedraAroundVertex.size()-1)
        createHexahedraAroundVertexArray();
    return m_hexahedraAroundVertex[i];
}

const MeshTopology::HexahedraAroundEdge& MeshTopology::getHexahedraAroundEdge(EdgeID i)
{
    if (!m_hexahedraAroundEdge.size() || i > m_hexahedraAroundEdge.size()-1)
        createHexahedraAroundEdgeArray();
    return m_hexahedraAroundEdge[i];
}

const MeshTopology::HexahedraAroundQuad& MeshTopology::getHexahedraAroundQuad(QuadID i)
{
    if (!m_hexahedraAroundQuad.size() || i > m_hexahedraAroundQuad.size()-1)
        createHexahedraAroundQuadArray();
    return m_hexahedraAroundQuad[i];
}




const vector< MeshTopology::EdgesInTriangle >& MeshTopology::getEdgesInTriangleArray()
{
    if(m_edgesInTriangle.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getEdgesInTriangleArray] EdgesInTriangle array is empty." << sendl;
#endif

        createEdgesInTriangleArray();
    }

    return m_edgesInTriangle;
}

const vector< MeshTopology::TrianglesAroundVertex >& MeshTopology::getTrianglesAroundVertexArray()
{
    if(m_trianglesAroundVertex.empty())	// this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getTrianglesAroundVertexArray] TrianglesAroundVertex array is empty." << sendl;
#endif

        createTrianglesAroundVertexArray();
    }

    return m_trianglesAroundVertex;
}

const vector< MeshTopology::TrianglesAroundEdge >& MeshTopology::getTrianglesAroundEdgeArray()
{
    if(m_trianglesAroundEdge.empty())	// this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getTrianglesAroundEdgeArray] TrianglesAroundEdge array is empty." << sendl;
#endif

        createTrianglesAroundEdgeArray();
    }

    return m_trianglesAroundEdge;
}




const vector< MeshTopology::EdgesInQuad >& MeshTopology::getEdgesInQuadArray()
{
    if(m_edgesInQuad.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getEdgesInQuadArray] EdgesInQuad array is empty." << sendl;
#endif

        createEdgesInQuadArray();
    }

    return m_edgesInQuad;
}

const vector< MeshTopology::QuadsAroundVertex >& MeshTopology::getQuadsAroundVertexArray()
{
    if(m_quadsAroundVertex.empty())	// this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getQuadsAroundVertexArray] QuadsAroundVertex array is empty." << sendl;
#endif

        createQuadsAroundVertexArray();
    }

    return m_quadsAroundVertex;
}

const vector< MeshTopology::QuadsAroundEdge >& MeshTopology::getQuadsAroundEdgeArray()
{
    if(m_quadsAroundEdge.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getQuadsAroundEdgeArray] QuadsAroundEdge array is empty." << sendl;
#endif

        createQuadsAroundEdgeArray();
    }

    return m_quadsAroundEdge;
}





const vector< MeshTopology::EdgesInTetrahedron >& MeshTopology::getEdgesInTetrahedronArray()
{
    if (m_edgesInTetrahedron.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getEdgesInTetrahedronArray] EdgesInTetrahedron array is empty." << sendl;
#endif

        createEdgesInTetrahedronArray();
    }

    return m_edgesInTetrahedron;
}

const vector< MeshTopology::TrianglesInTetrahedron >& MeshTopology::getTrianglesInTetrahedronArray()
{
    if (m_trianglesInTetrahedron.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getTrianglesInTetrahedronArray] TrianglesInTetrahedron array is empty." << sendl;
#endif

        createTrianglesInTetrahedronArray();
    }

    return m_trianglesInTetrahedron;
}

const vector< MeshTopology::TetrahedraAroundVertex >& MeshTopology::getTetrahedraAroundVertexArray()
{
    if (m_tetrahedraAroundVertex.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getTetrahedraAroundVertexArray] TetrahedraAroundVertex array is empty." << sendl;
#endif

        createTetrahedraAroundVertexArray();
    }

    return m_tetrahedraAroundVertex;
}

const vector< MeshTopology::TetrahedraAroundEdge >& MeshTopology::getTetrahedraAroundEdgeArray()
{
    if (m_tetrahedraAroundEdge.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getTetrahedraAroundEdgeArray] TetrahedraAroundEdge array is empty." << sendl;
#endif

        createTetrahedraAroundEdgeArray();
    }

    return m_tetrahedraAroundEdge;
}

const vector< MeshTopology::TetrahedraAroundTriangle >& MeshTopology::getTetrahedraAroundTriangleArray()
{
    if (m_tetrahedraAroundTriangle.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getTetrahedraAroundTriangleArray] TetrahedraAroundTriangle array is empty." << sendl;
#endif

        createTetrahedraAroundTriangleArray();
    }

    return m_tetrahedraAroundTriangle;
}




const vector< MeshTopology::EdgesInHexahedron >& MeshTopology::getEdgesInHexahedronArray()
{
    if (m_edgesInHexahedron.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getEdgesInHexahedronArray] EdgesInHexahedron array is empty." << sendl;
#endif

        createEdgesInHexahedronArray();
    }

    return m_edgesInHexahedron;
}

const vector< MeshTopology::QuadsInHexahedron >& MeshTopology::getQuadsInHexahedronArray()
{
    if (m_quadsInHexahedron.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getQuadsInHexahedronArray] QuadsInHexahedron array is empty." << sendl;
#endif

        createQuadsInHexahedronArray();
    }

    return m_quadsInHexahedron;
}

const vector< MeshTopology::HexahedraAroundVertex >& MeshTopology::getHexahedraAroundVertexArray()
{
    if (m_hexahedraAroundVertex.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getHexahedraAroundVertexArray] HexahedraAroundVertex array is empty." << sendl;
#endif

        createHexahedraAroundVertexArray();
    }

    return m_hexahedraAroundVertex;
}

const vector< MeshTopology::HexahedraAroundEdge >& MeshTopology::getHexahedraAroundEdgeArray()
{
    if (m_hexahedraAroundEdge.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getHexahedraAroundEdgeArray] HexahedraAroundEdge array is empty." << sendl;
#endif

        createHexahedraAroundEdgeArray();
    }

    return m_hexahedraAroundEdge;
}

const vector< MeshTopology::HexahedraAroundQuad >& MeshTopology::getHexahedraAroundQuadArray()
{
    if (m_hexahedraAroundQuad.empty()) // this method should only be called when the array exists.
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::getHexahedraAroundQuadArray] HexahedraAroundQuad array is empty." << sendl;
#endif

        createHexahedraAroundQuadArray();
    }

    return m_hexahedraAroundQuad;
}



int MeshTopology::getEdgeIndex(PointID v1, PointID v2)
{
    const EdgesAroundVertex &es1 = getEdgesAroundVertex(v1) ;
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
    //const vector< TrianglesAroundVertex > &tvs=getTrianglesAroundVertexArray();

    const vector<TriangleID> &set1=getTrianglesAroundVertex(v1);
    const vector<TriangleID> &set2=getTrianglesAroundVertex(v2);
    const vector<TriangleID> &set3=getTrianglesAroundVertex(v3);

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
    //const vector< QuadsAroundVertex > &qvs=getQuadsAroundVertexArray();

    const vector<QuadID> &set1=getQuadsAroundVertex(v1);
    const vector<QuadID> &set2=getQuadsAroundVertex(v2);
    const vector<QuadID> &set3=getQuadsAroundVertex(v3);
    const vector<QuadID> &set4=getQuadsAroundVertex(v4);

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

int MeshTopology::getTetrahedronIndex(PointID v1, PointID v2, PointID v3,  PointID v4)
{
    const vector<TetraID> &set1=getTetrahedraAroundVertex(v1);
    const vector<TetraID> &set2=getTetrahedraAroundVertex(v2);
    const vector<TetraID> &set3=getTetrahedraAroundVertex(v3);
    const vector<TetraID> &set4=getTetrahedraAroundVertex(v4);

    // The destination vector must be large enough to contain the result.
    vector<TetraID> out1(set1.size()+set2.size());
    vector<TetraID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    vector<TetraID> out2(set3.size()+out1.size());
    vector<TetraID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    vector<TetraID> out3(set4.size()+out2.size());
    vector<TetraID>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

    assert(out3.size()==0 || out3.size()==1);

    if (out3.size()==1)
        return (int) (out3[0]);
    else
        return -1;
}

int MeshTopology::getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4, PointID v5, PointID v6, PointID v7, PointID v8)
{
    const vector<HexaID> &set1=getTetrahedraAroundVertex(v1);
    const vector<HexaID> &set2=getTetrahedraAroundVertex(v2);
    const vector<HexaID> &set3=getTetrahedraAroundVertex(v3);
    const vector<HexaID> &set4=getTetrahedraAroundVertex(v4);
    const vector<HexaID> &set5=getTetrahedraAroundVertex(v5);
    const vector<HexaID> &set6=getTetrahedraAroundVertex(v6);
    const vector<HexaID> &set7=getTetrahedraAroundVertex(v7);
    const vector<HexaID> &set8=getTetrahedraAroundVertex(v8);

    // The destination vector must be large enough to contain the result.
    vector<HexaID> out1(set1.size()+set2.size());
    vector<HexaID>::iterator result1;
    result1 = std::set_intersection(set1.begin(),set1.end(),set2.begin(),set2.end(),out1.begin());
    out1.erase(result1,out1.end());

    vector<HexaID> out2(set3.size()+out1.size());
    vector<HexaID>::iterator result2;
    result2 = std::set_intersection(set3.begin(),set3.end(),out1.begin(),out1.end(),out2.begin());
    out2.erase(result2,out2.end());

    vector<HexaID> out3(set4.size()+out2.size());
    vector<HexaID>::iterator result3;
    result3 = std::set_intersection(set4.begin(),set4.end(),out2.begin(),out2.end(),out3.begin());
    out3.erase(result3,out3.end());

    vector<HexaID> out4(set5.size()+set3.size());
    vector<HexaID>::iterator result4;
    result4 = std::set_intersection(set5.begin(),set5.end(),set3.begin(),set3.end(),out4.begin());
    out4.erase(result4,out4.end());

    vector<HexaID> out5(set6.size()+out4.size());
    vector<HexaID>::iterator result5;
    result5 = std::set_intersection(set6.begin(),set6.end(),out4.begin(),out4.end(),out5.begin());
    out5.erase(result5,out5.end());

    vector<HexaID> out6(set7.size()+out5.size());
    vector<HexaID>::iterator result6;
    result6 = std::set_intersection(set7.begin(),set7.end(),out5.begin(),out5.end(),out6.begin());
    out6.erase(result6,out6.end());

    vector<HexaID> out7(set8.size()+out6.size());
    vector<HexaID>::iterator result7;
    result7 = std::set_intersection(set8.begin(),set8.end(),out6.begin(),out6.end(),out7.begin());
    out7.erase(result7,out7.end());

    assert(out7.size()==0 || out7.size()==1);

    if (out7.size()==1)
        return (int) (out7[0]);
    else
        return -1;
}

int MeshTopology::getVertexIndexInTriangle(const Triangle &t, PointID vertexIndex) const
{
    if (t[0]==vertexIndex)
        return 0;
    else if (t[1]==vertexIndex)
        return 1;
    else if (t[2]==vertexIndex)
        return 2;
    else
        return -1;
}

int MeshTopology::getEdgeIndexInTriangle(const EdgesInTriangle &t, EdgeID edgeIndex) const
{
    if (t[0]==edgeIndex)
        return 0;
    else if (t[1]==edgeIndex)
        return 1;
    else if (t[2]==edgeIndex)
        return 2;
    else
        return -1;
}

int MeshTopology::getVertexIndexInQuad(const Quad &t, PointID vertexIndex) const
{
    if(t[0]==vertexIndex)
        return 0;
    else if(t[1]==vertexIndex)
        return 1;
    else if(t[2]==vertexIndex)
        return 2;
    else if(t[3]==vertexIndex)
        return 3;
    else
        return -1;
}

int MeshTopology::getEdgeIndexInQuad(const EdgesInQuad &t, EdgeID edgeIndex) const
{
    if(t[0]==edgeIndex)
        return 0;
    else if(t[1]==edgeIndex)
        return 1;
    else if(t[2]==edgeIndex)
        return 2;
    else if(t[3]==edgeIndex)
        return 3;
    else
        return -1;
}

int MeshTopology::getVertexIndexInTetrahedron(const Tetra &t, PointID vertexIndex) const
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

int MeshTopology::getEdgeIndexInTetrahedron(const EdgesInTetrahedron &t, EdgeID edgeIndex) const
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

int MeshTopology::getTriangleIndexInTetrahedron(const TrianglesInTetrahedron &t, TriangleID triangleIndex) const
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

int MeshTopology::getVertexIndexInHexahedron(const Hexa &t, PointID vertexIndex) const
{
    if(t[0]==vertexIndex)
        return 0;
    else if(t[1]==vertexIndex)
        return 1;
    else if(t[2]==vertexIndex)
        return 2;
    else if(t[3]==vertexIndex)
        return 3;
    else if(t[4]==vertexIndex)
        return 4;
    else if(t[5]==vertexIndex)
        return 5;
    else if(t[6]==vertexIndex)
        return 6;
    else if(t[7]==vertexIndex)
        return 7;
    else
        return -1;
}

int MeshTopology::getEdgeIndexInHexahedron(const EdgesInHexahedron &t, EdgeID edgeIndex) const
{
    if(t[0]==edgeIndex)
        return 0;
    else if(t[1]==edgeIndex)
        return 1;
    else if(t[2]==edgeIndex)
        return 2;
    else if(t[3]==edgeIndex)
        return 3;
    else if(t[4]==edgeIndex)
        return 4;
    else if(t[5]==edgeIndex)
        return 5;
    else if(t[6]==edgeIndex)
        return 6;
    else if(t[7]==edgeIndex)
        return 7;
    else if(t[8]==edgeIndex)
        return 8;
    else if(t[9]==edgeIndex)
        return 9;
    else if(t[10]==edgeIndex)
        return 10;
    else if(t[11]==edgeIndex)
        return 11;
    else
        return -1;
}

int MeshTopology::getQuadIndexInHexahedron(const QuadsInHexahedron &t, QuadID quadIndex) const
{
    if(t[0]==quadIndex)
        return 0;
    else if(t[1]==quadIndex)
        return 1;
    else if(t[2]==quadIndex)
        return 2;
    else if(t[3]==quadIndex)
        return 3;
    else if(t[4]==quadIndex)
        return 4;
    else if(t[5]==quadIndex)
        return 5;
    else
        return -1;
}

MeshTopology::Edge MeshTopology::getLocalEdgesInTetrahedron (const unsigned int i) const
{
    assert(i<6);
    const unsigned int edgesInTetrahedronArray[6][2]= {{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
    return MeshTopology::Edge (edgesInTetrahedronArray[i][0], edgesInTetrahedronArray[i][1]);
}

MeshTopology::Edge MeshTopology::getLocalEdgesInHexahedron (const unsigned int i) const
{
    assert(i<12);
    const unsigned int edgesInHexahedronArray[12][2]= {{0,1},{0,3},{0,4},{1,2},{1,5},{2,3},{2,6},{3,7},{4,5},{4,7},{5,6},{6,7}};
    return MeshTopology::Edge (edgesInHexahedronArray[i][0], edgesInHexahedronArray[i][1]);
}


int MeshTopology::computeRelativeOrientationInTri(const unsigned int ind_p0, const unsigned int ind_p1, const unsigned int ind_t)
{
    const Triangle& t = getTriangles()[ind_t];
    int i = 0;
    while(i < (int)t.size())
    {
        if(ind_p0 == t[i])
            break;
        ++i;
    }

    if(i == (int)t.size()) //ind_p0 is not a PointID in the triangle ind_t
        return 0;

    if(ind_p1 == t[(i+1)%3]) //p0p1 has the same direction of t
        return 1;
    if(ind_p1 == t[(i+2)%3]) //p0p1 has the opposite direction of t
        return -1;

    return 0;
}

int MeshTopology::computeRelativeOrientationInQuad(const unsigned int ind_p0, const unsigned int ind_p1, const unsigned int ind_q)
{
    const Quad& q = getQuads()[ind_q];
    int i = 0;
    while(i < (int)q.size())
    {
        if(ind_p0 == q[i])
            break;
        ++i;
    }

    if(i == (int)q.size()) //ind_p0 is not a PointID in the quad ind_q
        return 0;

    if(ind_p1 == q[(i+1)%4]) //p0p1 has the same direction of q
        return 1;
    if(ind_p1 == q[(i+3)%4]) //p0p1 has the opposite direction of q
        return -1;

    return 0;
}

void MeshTopology::reOrientateTriangle(TriangleID id)
{
    if (id >= (unsigned int)this->getNbTriangles())
    {
#ifndef NDEBUG
        sout << "Warning. [MeshTopology::reOrientateTriangle] Triangle ID out of bounds." << sendl;
#endif
        return;
    }
    Triangle& tri = (*seqTriangles.beginEdit())[id];
    unsigned int tmp = tri[1];
    tri[1] = tri[2];
    tri[2] = tmp;
    seqTriangles.endEdit();

    return;
}

bool MeshTopology::hasPos() const
{
    return !seqPoints.getValue().empty();
}

SReal MeshTopology::getPX(int i) const
{
    return ((unsigned)i<seqPoints.getValue().size()?seqPoints.getValue()[i][0]:0.0);
}

SReal MeshTopology::getPY(int i) const
{
    return ((unsigned)i<seqPoints.getValue().size()?seqPoints.getValue()[i][1]:0.0);
}

SReal MeshTopology::getPZ(int i) const
{
    return ((unsigned)i<seqPoints.getValue().size()?seqPoints.getValue()[i][2]:0.0);
}

void MeshTopology::invalidate()
{

    validTetrahedra = false;
    validHexahedra = false;

    m_edgesAroundVertex.clear();
    m_edgesInTriangle.clear();
    m_edgesInQuad.clear();
    m_edgesInTetrahedron.clear();
    m_edgesInHexahedron.clear();
    m_trianglesAroundVertex.clear();
    m_trianglesAroundEdge.clear();
    m_trianglesInTetrahedron.clear();
    m_quadsAroundVertex.clear();
    m_quadsAroundEdge.clear();
    m_quadsInHexahedron.clear();
    m_tetrahedraAroundVertex.clear();
    m_tetrahedraAroundEdge.clear();
    m_tetrahedraAroundTriangle.clear();
    m_hexahedraAroundVertex.clear();
    m_hexahedraAroundEdge.clear();
    m_hexahedraAroundQuad.clear();
    ++revision;
    //sout << "MeshTopology::invalidate()"<<sendl;
}



void MeshTopology::updateHexahedra()
{
    if (!seqHexahedra.getValue().empty()) return; // hexahedra already defined
    // No 4D elements yet! ;)
}

void MeshTopology::updateTetrahedra()
{
    if (!seqTetrahedra.getValue().empty()) return; // tetrahedra already defined
    // No 4D elements yet! ;)
}




/// Get information about connexity of the mesh
/// @{

bool MeshTopology::checkConnexity()
{
    unsigned int nbr = 0;

    if (UpperTopology == core::topology::HEXAHEDRON)
        nbr = this->getNbHexahedra();
    else if (UpperTopology == core::topology::TETRAHEDRON)
        nbr = this->getNbTetrahedra();
    else if (UpperTopology == core::topology::QUAD)
        nbr = this->getNbQuads();
    else if (UpperTopology == core::topology::TRIANGLE)
        nbr = this->getNbTriangles();
    else
        nbr = this->getNbEdges();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [MeshTopology::checkConnexity] Can't compute connexity as some element are missing" << sendl;
#endif
        return false;
    }

    sofa::helper::vector <unsigned int> elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        serr << "Warning: in computing connexity, elements are missings. There is more than one connexe component." << sendl;
        return false;
    }

    return true;
}


unsigned int MeshTopology::getNumberOfConnectedComponent()
{
    unsigned int nbr = 0;

    if (UpperTopology == core::topology::HEXAHEDRON)
        nbr = this->getNbHexahedra();
    else if (UpperTopology == core::topology::TETRAHEDRON)
        nbr = this->getNbTetrahedra();
    else if (UpperTopology == core::topology::QUAD)
        nbr = this->getNbQuads();
    else if (UpperTopology == core::topology::TRIANGLE)
        nbr = this->getNbTriangles();
    else
        nbr = this->getNbEdges();

    if (nbr == 0)
    {
#ifndef NDEBUG
        serr << "Warning. [MeshTopology::getNumberOfConnectedComponent] Can't compute connexity as there are no elements" << sendl;
#endif
        return 0;
    }

    sofa::helper::vector <unsigned int> elemAll = this->getConnectedElement(0);
    unsigned int cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        unsigned int other_ID = (unsigned int)elemAll.size();

        for (unsigned int i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_ID = i;
                break;
            }

        sofa::helper::vector <unsigned int> elemTmp = this->getConnectedElement(other_ID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const sofa::helper::vector <unsigned int> MeshTopology::getConnectedElement(unsigned int elem)
{
    unsigned int nbr = 0;

    if (UpperTopology == core::topology::HEXAHEDRON)
        nbr = this->getNbHexahedra();
    else if (UpperTopology == core::topology::TETRAHEDRON)
        nbr = this->getNbTetrahedra();
    else if (UpperTopology == core::topology::QUAD)
        nbr = this->getNbQuads();
    else if (UpperTopology == core::topology::TRIANGLE)
        nbr = this->getNbTriangles();
    else
        nbr = this->getNbEdges();

    sofa::helper::vector <unsigned int> elemAll;
    sofa::helper::vector <unsigned int> elemOnFront, elemPreviousFront, elemNextFront;
    bool end = false;
    unsigned int cpt = 0;

    // init algo
    elemAll.push_back(elem);
    elemOnFront.push_back(elem);
    elemPreviousFront.clear();
    cpt++;

    while (!end && cpt < nbr)
    {
        // First Step - Create new region
        elemNextFront = this->getElementAroundElements(elemOnFront); // for each elementID on the propagation front

        // Second Step - Avoid backward direction
        for (unsigned int i = 0; i<elemNextFront.size(); ++i)
        {
            bool find = false;
            unsigned int id = elemNextFront[i];

            for (unsigned int j = 0; j<elemAll.size(); ++j)
                if (id == elemAll[j])
                {
                    find = true;
                    break;
                }

            if (!find)
            {
                elemAll.push_back(id);
                elemPreviousFront.push_back(id);
            }
        }

        // cpt for connexity
        cpt += (unsigned int)elemPreviousFront.size();

        if (elemPreviousFront.empty())
        {
            end = true;
#ifndef NDEBUG
            serr << "Loop for computing connexity has reach end." << sendl;
#endif
        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }

    return elemAll;
}


const sofa::helper::vector <unsigned int> MeshTopology::getElementAroundElement(unsigned int elem)
{
    sofa::helper::vector <unsigned int> elems;
    unsigned int nbr = 0;

    if (UpperTopology == core::topology::HEXAHEDRON)
    {
        nbr = 8;
        if(!this->m_hexahedraAroundVertex.empty())
            createHexahedraAroundVertexArray();
    }
    else if (UpperTopology == core::topology::TETRAHEDRON)
    {
        nbr = 4;
        if(!this->m_tetrahedraAroundVertex.empty())
            createTetrahedraAroundVertexArray();
    }
    else if (UpperTopology == core::topology::QUAD)
    {
        nbr = 4;
        if(!this->m_quadsAroundVertex.empty())
            createQuadsAroundVertexArray();
    }
    else if (UpperTopology == core::topology::TRIANGLE)
    {
        nbr = 3;
        if(!this->m_trianglesAroundVertex.empty())
            createTrianglesAroundVertexArray();
    }
    else
    {
        nbr = 2;
        if(!this->m_edgesAroundVertex.empty())
            createEdgesAroundVertexArray();
    }


    //Triangle the_tri = this->getTriangle(elem);

    for(unsigned int i = 0; i<nbr; ++i) // for each node of the triangle
    {
        sofa::helper::vector <unsigned int> elemAV;

        if (UpperTopology == core::topology::HEXAHEDRON)
            elemAV = this->getHexahedraAroundVertex(getHexahedron(elem)[i]);
        else if (UpperTopology == core::topology::TETRAHEDRON)
            elemAV = this->getTetrahedraAroundVertex(getTetrahedron(elem)[i]);
        else if (UpperTopology == core::topology::QUAD)
            elemAV = this->getQuadsAroundVertex(getQuad(elem)[i]);
        else if (UpperTopology == core::topology::TRIANGLE)
            elemAV = this->getTrianglesAroundVertex(getTriangle(elem)[i]);
        else
            elemAV = this->getEdgesAroundVertex(getEdge(elem)[i]);


        for (unsigned int j = 0; j<elemAV.size(); ++j) // for each element around the node
        {
            bool find = false;
            unsigned int id = elemAV[j];

            if (id == elem)
                continue;

            for (unsigned int k = 0; k<elems.size(); ++k) // check no redundancy
                if (id == elems[k])
                {
                    find = true;
                    break;
                }

            if (!find)
                elems.push_back(id);
        }
    }

    return elems;
}


const sofa::helper::vector <unsigned int> MeshTopology::getElementAroundElements(sofa::helper::vector <unsigned int> elems)
{
    sofa::helper::vector <unsigned int> elemAll;
    sofa::helper::vector <unsigned int> elemTmp;

    if (UpperTopology == core::topology::HEXAHEDRON)
    {
        if(!this->m_hexahedraAroundVertex.empty())
            createHexahedraAroundVertexArray();
    }
    else if (UpperTopology == core::topology::TETRAHEDRON)
    {
        if(!this->m_tetrahedraAroundVertex.empty())
            createTetrahedraAroundVertexArray();
    }
    else if (UpperTopology == core::topology::QUAD)
    {
        if(!this->m_quadsAroundVertex.empty())
            createQuadsAroundVertexArray();
    }
    else if (UpperTopology == core::topology::TRIANGLE)
    {
        if(!this->m_trianglesAroundVertex.empty())
            createTrianglesAroundVertexArray();
    }
    else
    {
        if(!this->m_edgesAroundVertex.empty())
            createEdgesAroundVertexArray();
    }


    for (unsigned int i = 0; i <elems.size(); ++i) // for each elementID of input vector
    {
        sofa::helper::vector <unsigned int> elemTmp2 = this->getElementAroundElement(elems[i]);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (unsigned int i = 0; i<elemTmp.size(); ++i) // for each elementID found
    {
        bool find = false;
        unsigned int id = elemTmp[i];

        for (unsigned int j = 0; j<elems.size(); ++j) // check no redundancy with input vector
            if (id == elems[j])
            {
                find = true;
                break;
            }

        if (!find)
        {
            for (unsigned int j = 0; j<elemAll.size(); ++j) // check no redundancy in output vector
                if (id == elemAll[j])
                {
                    find = true;
                    break;
                }
        }

        if (!find)
            elemAll.push_back(id);
    }


    return elemAll;
}

/// @}

SReal MeshTopology::getPosX(int i) const
{
    return ((unsigned)i<seqPoints.getValue().size()?seqPoints.getValue()[i][0]:0.0);
}

SReal MeshTopology::getPosY(int i) const
{
    return ((unsigned)i<seqPoints.getValue().size()?seqPoints.getValue()[i][1]:0.0);
}

SReal MeshTopology::getPosZ(int i) const
{
    return ((unsigned)i<seqPoints.getValue().size()?seqPoints.getValue()[i][2]:0.0);
}

void MeshTopology::draw(const core::visual::VisualParams* vparams)
{
    // Draw Edges
    if(_drawEdges.getValue())
    {
        std::vector<defaulttype::Vector3> pos;
        pos.reserve(this->getNbEdges()*2u);
        for (int i=0; i<getNbEdges(); i++)
        {
            const Edge& c = getEdge(i);
            pos.push_back(defaulttype::Vector3(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0])));
            pos.push_back(defaulttype::Vector3(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1])));
        }
        vparams->drawTool()->drawLines(pos, 1.0f, defaulttype::Vec4f(0.4f,1.0f,0.3f,1.0f));
    }

    //Draw Triangles
    if(_drawTriangles.getValue())
    {
        std::vector<defaulttype::Vector3> pos;
        pos.reserve(this->getNbTriangles()*3u);
        for (int i=0; i<getNbTriangles(); i++)
        {
            const Triangle& c = getTriangle(i);
            pos.push_back(defaulttype::Vector3(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0])));
            pos.push_back(defaulttype::Vector3(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1])));
            pos.push_back(defaulttype::Vector3(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2])));
        }
        vparams->drawTool()->drawTriangles(pos, defaulttype::Vec4f(0.4f,1.0f,0.3f,1.0f));
    }

    //Draw Quads
    if(_drawQuads.getValue())
    {
        std::vector<defaulttype::Vector3> pos;
        pos.reserve(this->getNbQuads()*4u);
        for (int i=0; i<getNbQuads(); i++)
        {
            const Quad& c = getQuad(i);
            pos.push_back(defaulttype::Vector3(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0])));
            pos.push_back(defaulttype::Vector3(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1])));
            pos.push_back(defaulttype::Vector3(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2])));
            pos.push_back(defaulttype::Vector3(getPosX(c[3]), getPosY(c[3]), getPosZ(c[3])));
        }
        vparams->drawTool()->drawQuads(pos, defaulttype::Vec4f(0.4f,1.0f,0.3f,1.0f));
    }

    //Draw Hexahedron
    if (_drawHexa.getValue())
    {
        std::vector<defaulttype::Vector3> pos1;
        std::vector<defaulttype::Vector3> pos2;
        pos1.reserve(this->getNbHexahedra()*8u);
        pos2.reserve(this->getNbHexahedra()*8u);
        for (int i=0; i<getNbHexahedra(); i++)
        {
            const Hexa& c = getHexahedron(i);
            pos1.push_back(defaulttype::Vector3(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[3]), getPosY(c[3]), getPosZ(c[3])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[4]), getPosY(c[4]), getPosZ(c[4])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[5]), getPosY(c[5]), getPosZ(c[5])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[6]), getPosY(c[6]), getPosZ(c[6])));
            pos1.push_back(defaulttype::Vector3(getPosX(c[7]), getPosY(c[7]), getPosZ(c[7])));

            pos2.push_back(defaulttype::Vector3(getPosX(c[3]), getPosY(c[3]), getPosZ(c[3])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[7]), getPosY(c[7]), getPosZ(c[7])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[6]), getPosY(c[6]), getPosZ(c[6])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[4]), getPosY(c[4]), getPosZ(c[4])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1])));
            pos2.push_back(defaulttype::Vector3(getPosX(c[5]), getPosY(c[5]), getPosZ(c[5])));
        }
        vparams->drawTool()->drawQuads(pos1, defaulttype::Vec4f(0.4f,1.0f,0.3f,1.0f));
        vparams->drawTool()->drawLines(pos2, 1.0f, defaulttype::Vec4f(0.4f,1.0f,0.3f,1.0f));
    }

    // Draw Tetra
    if(_drawTetra.getValue())
    {
        std::vector<defaulttype::Vector3> pos;
        pos.reserve(this->getNbTetrahedra()*12u);
        for (int i=0; i<getNbTetras(); i++)
        {
            const Tetra& t = getTetra(i);
            pos.push_back(defaulttype::Vector3(getPosX(t[0]), getPosY(t[0]), getPosZ(t[0])));
            pos.push_back(defaulttype::Vector3(getPosX(t[1]), getPosY(t[1]), getPosZ(t[1])));

            pos.push_back(defaulttype::Vector3(getPosX(t[0]), getPosY(t[0]), getPosZ(t[0])));
            pos.push_back(defaulttype::Vector3(getPosX(t[2]), getPosY(t[2]), getPosZ(t[2])));

            pos.push_back(defaulttype::Vector3(getPosX(t[0]), getPosY(t[0]), getPosZ(t[0])));
            pos.push_back(defaulttype::Vector3(getPosX(t[3]), getPosY(t[3]), getPosZ(t[3])));

            pos.push_back(defaulttype::Vector3(getPosX(t[1]), getPosY(t[1]), getPosZ(t[1])));
            pos.push_back(defaulttype::Vector3(getPosX(t[2]), getPosY(t[2]), getPosZ(t[2])));

            pos.push_back(defaulttype::Vector3(getPosX(t[1]), getPosY(t[1]), getPosZ(t[1])));
            pos.push_back(defaulttype::Vector3(getPosX(t[3]), getPosY(t[3]), getPosZ(t[3])));

            pos.push_back(defaulttype::Vector3(getPosX(t[2]), getPosY(t[2]), getPosZ(t[2])));
            pos.push_back(defaulttype::Vector3(getPosX(t[3]), getPosY(t[3]), getPosZ(t[3])));
        }
        vparams->drawTool()->drawLines(pos, 1.0f, defaulttype::Vec4f(1.0f,0.0f,0.0f,1.0f));
    }

}

} // namespace topology

} // namespace component

} // namespace sofa
