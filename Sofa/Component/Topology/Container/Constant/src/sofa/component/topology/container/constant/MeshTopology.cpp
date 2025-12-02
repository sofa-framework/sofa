/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/component/topology/container/constant/MeshTopology.h>

#include <sofa/core/topology/Topology.h>
#include <sofa/helper/visual/DrawTool.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <algorithm>

namespace sofa::component::topology::container::constant
{

using type::vector;
using sofa::core::topology::edgesInTetrahedronArray;
using sofa::core::topology::edgesInHexahedronArray;

MeshTopology::EdgeUpdate::EdgeUpdate(MeshTopology* t)
    :PrimitiveUpdate(t)
{
    if( topology->hasVolume() )
    {
        addInput(&topology->d_seqHexahedra);
        addInput(&topology->d_seqTetrahedra);
        addOutput(&topology->d_seqEdges);
        setDirtyValue();
    }
    else if( topology->hasSurface() )
    {
        addInput(&topology->d_seqTriangles);
        addInput(&topology->d_seqQuads);
        addOutput(&topology->d_seqEdges);
        setDirtyValue();
    }

}

void MeshTopology::EdgeUpdate::doUpdate()
{
    if(topology->hasVolume() ) updateFromVolume();
    else if(topology->hasSurface()) updateFromSurface();
}

void MeshTopology::EdgeUpdate::updateFromVolume()
{
    typedef MeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef MeshTopology::SeqHexahedra  SeqHexahedra;
    typedef MeshTopology::SeqEdges     SeqEdges;

    SeqEdges& seqEdges = *topology->d_seqEdges.beginEdit();
    seqEdges.clear();
    std::map<Edge,unsigned int> edgeMap;
    unsigned int edgeIndex;

    const SeqTetrahedra& tetrahedra = topology->getTetrahedra(); // do not use d_seqTetrahedra directly as it might not be up-to-date
    for (const auto& t : tetrahedra)
    {
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        for (const auto j : edgesInTetrahedronArray)
        {
            unsigned int v1 = t[j[0]];
            unsigned int v2 = t[j[1]];
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

    const SeqHexahedra& hexahedra = topology->getHexahedra(); // do not use d_seqHexahedra directly as it might not be up-to-date
    // create a temporary map to find redundant edges

    /// create the m_edge array at the same time than it fills the m_edgesInHexahedron array
    for (const auto& h : hexahedra)
    {
        std::map<Edge,unsigned int>::iterator ite;
        Edge e;
        for (const auto j : edgesInHexahedronArray)
        {
            unsigned int v1 = h[j[0]];
            unsigned int v2 = h[j[1]];
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
    topology->d_seqEdges.endEdit();
}

void MeshTopology::EdgeUpdate::updateFromSurface()
{
    typedef MeshTopology::SeqTriangles SeqTriangles;
    typedef MeshTopology::SeqQuads     SeqQuads;
    typedef MeshTopology::SeqEdges     SeqEdges;

    std::map<Edge,unsigned int> edgeMap;
    unsigned int edgeIndex;
    SeqEdges& seqEdges = *topology->d_seqEdges.beginEdit();
    seqEdges.clear();
    const SeqTriangles& triangles = topology->getTriangles(); // do not use d_seqTriangles directly as it might not be up-to-date
    for (const auto& t : triangles)
    {
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
                // To be similar to TriangleSetTopologyContainer::createEdgeSetArray
                //d_seqEdges.push_back(e); Changed to have oriented edges on the border of the triangulation.
                seqEdges.push_back(Edge(v1, v2));
            }
//            else
//            {
//                edgeIndex=(*ite).second;
//            }
            //m_edgesInTriangle[i][j]=edgeIndex;
        }
    }

    const SeqQuads& quads = topology->getQuads(); // do not use d_seqQuads directly as it might not be up-to-date
    for (const auto& t : quads)
    {
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

    topology->d_seqEdges.endEdit();
}


MeshTopology::TriangleUpdate::TriangleUpdate(MeshTopology *t)
    :PrimitiveUpdate(t)
{
    addInput(&topology->d_seqTetrahedra);
    addOutput(&topology->d_seqTriangles);
    setDirtyValue();
}


void MeshTopology::TriangleUpdate::doUpdate()
{
    typedef MeshTopology::SeqTetrahedra SeqTetrahedra;
    typedef MeshTopology::SeqTriangles SeqTriangles;
    const SeqTetrahedra& tetrahedra = topology->getTetrahedra();
    SeqTriangles& seqTriangles = *topology->d_seqTriangles.beginEdit();
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
        const Tetra &t=topology->d_seqTetrahedra.getValue()[i];
        for (const auto j : sofa::core::topology::trianglesOrientationInTetrahedronArray)
        {
            for (PointID k=0; k<3; ++k)
                v[k] = t[j[k]];

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
        }
    }

    topology->d_seqTriangles.endEdit();
}

MeshTopology::QuadUpdate::QuadUpdate(MeshTopology *t)
    :PrimitiveUpdate(t)
{
    addInput(&topology->d_seqHexahedra);
    addOutput(&topology->d_seqQuads);
    setDirtyValue();
}

void MeshTopology::QuadUpdate::doUpdate()
{
    typedef MeshTopology::SeqHexahedra SeqHexahedra;
    typedef MeshTopology::SeqQuads SeqQuads;

    SeqQuads& seqQuads = *topology->d_seqQuads.beginEdit();
    seqQuads.clear();

    if (topology->getNbHexahedra()==0) return; // no hexahedra to extract edges from

    const SeqHexahedra& hexahedra = topology->getHexahedra(); // do not use d_seqQuads directly as it might not be up-to-date

    // create a temporary map to find redundant quads
    std::map<Quad,unsigned int> quadMap;
    std::map<Quad,unsigned int>::iterator itt;
    Quad qu;
    unsigned int v[4],val;
    unsigned int quadIndex;
    /// create the m_edge array at the same time than it fills the m_edgesInHexahedron array
    for (const auto& h : hexahedra)
    {
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
            quadIndex=(unsigned int)topology->d_seqQuads.getValue().size();
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

    topology->d_seqQuads.endEdit();
}

using namespace sofa::defaulttype;
using core::topology::BaseMeshTopology;

void registerMeshTopology(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Generic constant topology loaded from a mesh file.")
        .add< MeshTopology >());
}

MeshTopology::MeshTopology()
    : d_seqPoints(initData(&d_seqPoints, "position", "List of point positions"))
    , d_seqEdges(initData(&d_seqEdges, "edges", "List of edge indices"))
    , d_seqTriangles(initData(&d_seqTriangles, "triangles", "List of triangle indices"))
    , d_seqQuads(initData(&d_seqQuads, "quads", "List of quad indices"))
    , d_seqTetrahedra(initData(&d_seqTetrahedra, "tetrahedra", "List of tetrahedron indices"))
    , d_seqHexahedra(initData(&d_seqHexahedra, "hexahedra", "List of hexahedron indices"))
    , d_seqPrisms(initData(&d_seqPrisms, "prisms", "List of prisms indices"))
    , d_seqPyramids(initData(&d_seqPyramids, "pyramids", "List of pyramids indices"))
    , d_seqUVs(initData(&d_seqUVs, "uv", "List of uv coordinates"))
    , d_computeAllBuffers(initData(&d_computeAllBuffers, false, "computeAllBuffers", "Option to compute all crossed topology buffers at init. False by default"))
    , nbPoints(0)
    , validTetrahedra(false), validHexahedra(false)
    , revision(0)
    , d_drawEdges(initData(&d_drawEdges, false, "drawEdges", "if true, draw the topology Edges"))
    , d_drawTriangles(initData(&d_drawTriangles, false, "drawTriangles", "if true, draw the topology Triangles"))
    , d_drawQuads(initData(&d_drawQuads, false, "drawQuads", "if true, draw the topology Quads"))
    , d_drawTetra(initData(&d_drawTetra, false, "drawTetrahedra", "if true, draw the topology Tetrahedra"))
    , d_drawHexa(initData(&d_drawHexa, false, "drawHexahedra", "if true, draw the topology hexahedra"))
{
    m_upperElementType = sofa::geometry::ElementType::EDGE;
    addAlias(&d_seqPoints, "points");
    addAlias(&d_seqEdges, "lines");
    addAlias(&d_seqTetrahedra, "tetras");
    addAlias(&d_seqHexahedra, "hexas");
    addAlias(&d_seqUVs, "texcoords");
}

void MeshTopology::init()
{
    BaseMeshTopology::init();

    const auto hexahedra = sofa::helper::getReadAccessor(d_seqHexahedra);
    const auto tetrahedra = sofa::helper::getReadAccessor(d_seqTetrahedra);
    const auto quads = sofa::helper::getReadAccessor(d_seqQuads);
    const auto triangles = sofa::helper::getReadAccessor(d_seqTriangles);
    const auto edges = sofa::helper::getReadAccessor(d_seqEdges);

    // looking for upper topology
    if (!hexahedra.empty())
        m_upperElementType = geometry::ElementType::HEXAHEDRON;
    else if (!tetrahedra.empty())
        m_upperElementType = sofa::geometry::ElementType::TETRAHEDRON;
    else if (!quads.empty())
        m_upperElementType = sofa::geometry::ElementType::QUAD;
    else if (!triangles.empty())
        m_upperElementType = sofa::geometry::ElementType::TRIANGLE;
    else if (!edges.empty())
        m_upperElementType = sofa::geometry::ElementType::EDGE;
    else
        m_upperElementType = sofa::geometry::ElementType::POINT;

    // If true, will compute all crossed element buffers such as triangleAroundEdges, EdgesIntriangle, etc.
    if (d_computeAllBuffers.getValue())
    {
        computeCrossElementBuffers();
    }

    // compute the number of points, if the topology is charged from the scene or if it was loaded from a MeshLoader without any points data.
    if (nbPoints==0)
    {
        unsigned int n = 0;
        const auto countPoints = [&n](const auto& seqElements)
        {
            for (const auto& element : seqElements)
            {
                for (const auto pointId : element)
                {
                    if (n <= pointId)
                    {
                        n = 1 + pointId;
                    }
                }
            }
        };

        countPoints(edges);
        countPoints(triangles);
        countPoints(quads);
        countPoints(tetrahedra);
        countPoints(hexahedra);

        nbPoints = n;
    }


    if(edges.empty() )
    {
        if(d_seqEdges.getParent() != nullptr )
        {
            d_seqEdges.delInput(d_seqEdges.getParent());
        }
        const EdgeUpdate::SPtr edgeUpdate = sofa::core::objectmodel::New<EdgeUpdate>(this);
        edgeUpdate->setName("edgeUpdate");
        this->addSlave(edgeUpdate);
    }
    if(triangles.empty() )
    {
        if(d_seqTriangles.getParent() != nullptr)
        {
            d_seqTriangles.delInput(d_seqTriangles.getParent());
        }
        const TriangleUpdate::SPtr triangleUpdate = sofa::core::objectmodel::New<TriangleUpdate>(this);
        triangleUpdate->setName("triangleUpdate");
        this->addSlave(triangleUpdate);
    }
    if(quads.empty() )
    {
        if(d_seqQuads.getParent() != nullptr )
        {
            d_seqQuads.delInput(d_seqQuads.getParent());
        }
        const QuadUpdate::SPtr quadUpdate = sofa::core::objectmodel::New<QuadUpdate>(this);
        quadUpdate->setName("quadUpdate");
        this->addSlave(quadUpdate);
    }
}

void MeshTopology::computeCrossElementBuffers()
{
    const auto hexahedra = sofa::helper::getReadAccessor(d_seqHexahedra);
    const auto tetrahedra = sofa::helper::getReadAccessor(d_seqTetrahedra);
    const auto quads = sofa::helper::getReadAccessor(d_seqQuads);
    const auto triangles = sofa::helper::getReadAccessor(d_seqTriangles);
    const auto edges = sofa::helper::getReadAccessor(d_seqEdges);

    if (!hexahedra.empty()) // Create hexahedron cross element buffers.
    {
        createHexahedraAroundVertexArray();

        if (!quads.empty())
        {
            createQuadsInHexahedronArray();
            createHexahedraAroundQuadArray();
        }

        if (!edges.empty())
        {
            createEdgesInHexahedronArray();
            createHexahedraAroundEdgeArray();
        }
    }
    if (!tetrahedra.empty()) // Create tetrahedron cross element buffers.
    {
        createTetrahedraAroundVertexArray();

        if (!triangles.empty())
        {
            createTrianglesInTetrahedronArray();
            createTetrahedraAroundTriangleArray();
        }

        if (!edges.empty())
        {
            createEdgesInTetrahedronArray();
            createTetrahedraAroundEdgeArray();
        }
    }
    if (!quads.empty()) // Create triangle cross element buffers.
    {
        createQuadsAroundVertexArray();

        if (!edges.empty())
        {
            createEdgesInQuadArray();
            createQuadsAroundEdgeArray();
        }
    }
    if (!triangles.empty()) // Create triangle cross element buffers.
    {
        createTrianglesAroundVertexArray();

        if (!edges.empty())
        {
            createEdgesInTriangleArray();
            createTrianglesAroundEdgeArray();
        }
    }
    if (!edges.empty())
    {
        createEdgesAroundVertexArray();
    }
}

void MeshTopology::clear()
{
    nbPoints = 0;

    helper::getWriteOnlyAccessor(d_seqPoints).clear();
    helper::getWriteOnlyAccessor(d_seqEdges).clear();
    helper::getWriteOnlyAccessor(d_seqTriangles).clear();
    helper::getWriteOnlyAccessor(d_seqQuads).clear();
    helper::getWriteOnlyAccessor(d_seqTetrahedra).clear();
    helper::getWriteOnlyAccessor(d_seqHexahedra).clear();
    helper::getWriteOnlyAccessor(d_seqUVs).clear();

    invalidate();
}


void MeshTopology::addPoint(SReal px, SReal py, SReal pz)
{
    d_seqPoints.beginEdit()->push_back(type::Vec3((SReal)px, (SReal)py, (SReal)pz));
    d_seqPoints.endEdit();
    if (d_seqPoints.getValue().size() > nbPoints)
        nbPoints = (int)d_seqPoints.getValue().size();
}

void MeshTopology::addEdge( Index a, Index b )
{
    d_seqEdges.beginEdit()->push_back(Edge(a, b));
    d_seqEdges.endEdit();
    if (a >= nbPoints) nbPoints = a+1;
    if (b >= nbPoints) nbPoints = b+1;
}

void MeshTopology::addTriangle( Index a, Index b, Index c )
{
    d_seqTriangles.beginEdit()->push_back(Triangle(a, b, c) );
    d_seqTriangles.endEdit();
    if (a >= nbPoints) nbPoints = a+1;
    if (b >= nbPoints) nbPoints = b+1;
    if (c >= nbPoints) nbPoints = c+1;
}

void MeshTopology::addQuad(Index a, Index b, Index c, Index d)
{
    d_seqQuads.beginEdit()->push_back(Quad(a, b, c, d));
    d_seqQuads.endEdit();
    if (a >= nbPoints) nbPoints = a+1;
    if (b >= nbPoints) nbPoints = b+1;
    if (c >= nbPoints) nbPoints = c+1;
    if (d >= nbPoints) nbPoints = d+1;
}

void MeshTopology::addTetra( Index a, Index b, Index c, Index d )
{
    d_seqTetrahedra.beginEdit()->push_back(Tetra(a, b, c, d) );
    d_seqTetrahedra.endEdit();
    if (a >= nbPoints) nbPoints = a+1;
    if (b >= nbPoints) nbPoints = b+1;
    if (c >= nbPoints) nbPoints = c+1;
    if (d >= nbPoints) nbPoints = d+1;
}

void MeshTopology::addHexa(Index p1, Index p2, Index p3, Index p4, Index p5, Index p6, Index p7, Index p8)
{

    d_seqHexahedra.beginEdit()->push_back(Hexa(p1, p2, p3, p4, p5, p6, p7, p8));

    d_seqHexahedra.endEdit();
    if (p1 >= nbPoints) nbPoints = p1+1;
    if (p2 >= nbPoints) nbPoints = p2+1;
    if (p3 >= nbPoints) nbPoints = p3+1;
    if (p4 >= nbPoints) nbPoints = p4+1;
    if (p5 >= nbPoints) nbPoints = p5+1;
    if (p6 >= nbPoints) nbPoints = p6+1;
    if (p7 >= nbPoints) nbPoints = p7+1;
    if (p8 >= nbPoints) nbPoints = p8+1;
}

void MeshTopology::addUV(SReal u, SReal v)
{
    d_seqUVs.beginEdit()->push_back(type::Vec<2,SReal>((SReal)u, (SReal)v));
    d_seqUVs.endEdit();
    if (d_seqUVs.getValue().size() > nbPoints)
        nbPoints = (int)d_seqUVs.getValue().size();
}

const MeshTopology::SeqEdges& MeshTopology::getEdges()
{
    return d_seqEdges.getValue();
}

const MeshTopology::SeqTriangles& MeshTopology::getTriangles()
{
    return d_seqTriangles.getValue();
}

const MeshTopology::SeqQuads& MeshTopology::getQuads()
{
    return d_seqQuads.getValue();
}

const MeshTopology::SeqTetrahedra& MeshTopology::getTetrahedra()
{
    if (!validTetrahedra)
    {
        updateTetrahedra();
        validTetrahedra = true;
    }
    return d_seqTetrahedra.getValue();
}

const MeshTopology::SeqHexahedra& MeshTopology::getHexahedra()
{
    if (!validHexahedra)
    {
        updateHexahedra();
        validHexahedra = true;
    }
    return d_seqHexahedra.getValue();
}

const BaseMeshTopology::SeqPrisms& MeshTopology::getPrisms()
{
    return d_seqPrisms.getValue();
}

const BaseMeshTopology::SeqPyramids& MeshTopology::getPyramids()
{
    return d_seqPyramids.getValue();
}

const MeshTopology::SeqUV& MeshTopology::getUVs()
{
    return d_seqUVs.getValue();
}

Size MeshTopology::getNbPoints() const
{
    return nbPoints;
}

void MeshTopology::setNbPoints(Size n)
{
    nbPoints = n;
}

Size MeshTopology::getNbUVs()
{
    return Size(getUVs().size());
}

const MeshTopology::UV MeshTopology::getUV(Index i)
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
            m_edgesAroundVertex[ edges[i][1] ].push_back( i );
        }
    }
}

void MeshTopology::createEdgesInTriangleArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use d_seqEdges directly as it might not be up-to-date
    const SeqTriangles& triangles = getTriangles(); // do not use d_seqTriangles directly as it might not be up-to-date
    m_edgesInTriangle.clear();
    m_edgesInTriangle.resize(triangles.size());
    for (unsigned int i = 0; i < triangles.size(); ++i)
    {
        const Triangle &t=triangles[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<3; ++j)
        {
            const EdgeID edgeIndex=getEdgeIndex(t[(j+1)%3],t[(j+2)%3]);
            assert(edgeIndex != InvalidID);
            m_edgesInTriangle[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createEdgesInQuadArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use d_seqEdges directly as it might not be up-to-date
    const SeqQuads& quads = getQuads(); // do not use d_seqQuads directly as it might not be up-to-date
    m_edgesInQuad.clear();
    m_edgesInQuad.resize(quads.size());
    for (unsigned int i = 0; i < quads.size(); ++i)
    {
        const Quad &t=quads[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<4; ++j)
        {
            const EdgeID edgeIndex = getEdgeIndex(t[(j+1)%4],t[(j+2)%4]);
            assert(edgeIndex != InvalidID);
            m_edgesInQuad[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createEdgesInTetrahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use d_seqEdges directly as it might not be up-to-date
    const SeqTetrahedra& tetrahedra = getTetrahedra(); // do not use d_seqTetrahedra directly as it might not be up-to-date
    m_edgesInTetrahedron.clear();
    m_edgesInTetrahedron.resize(tetrahedra.size());

    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        const Tetra &t=tetrahedra[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<6; ++j)
        {
            const EdgeID edgeIndex = getEdgeIndex(t[edgesInTetrahedronArray[j][0]], t[edgesInTetrahedronArray[j][1]]);
            assert(edgeIndex != InvalidID);
            m_edgesInTetrahedron[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createEdgesInHexahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use d_seqEdges directly as it might not be up-to-date
    //getEdges();
    const SeqHexahedra& hexahedra = getHexahedra(); // do not use d_seqHexahedra directly as it might not be up-to-date
    m_edgesInHexahedron.clear();
    m_edgesInHexahedron.resize(hexahedra.size());

    for (unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        const Hexa &h=hexahedra[i];
        // adding edge i in the edge shell of both points
        for (unsigned int j=0; j<12; ++j)
        {
            const EdgeID edgeIndex = getEdgeIndex(h[edgesInHexahedronArray[j][0]], h[edgesInHexahedronArray[j][1]]);
            assert(edgeIndex != InvalidID);
            m_edgesInHexahedron[i][j]=edgeIndex;
        }
    }
}

void MeshTopology::createTrianglesAroundVertexArray ()
{
    const SeqTriangles& triangles = getTriangles(); // do not use d_seqTriangles directly as it might not be up-to-date
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
            for (const unsigned int j = 0; i < m_orientedEdgesAroundVertex[i].size(); ++i)
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
    const SeqTriangles& triangles = getTriangles(); // do not use d_seqTriangles directly as it might not be up-to-date
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
            if (d_seqEdges.getValue()[tea[i][j]][0] == t[(j + 1) % 3])
                m_trianglesAroundEdge[ tea[i][j] ].insert(m_trianglesAroundEdge[ tea[i][j] ].begin(), i); // triangle is on the left of the edge
            else
                m_trianglesAroundEdge[ tea[i][j] ].push_back( i ); // triangle is on the right of the edge
        }
    }
}

void MeshTopology::createTrianglesInTetrahedronArray ()
{
    //const SeqEdges& edges = getEdges(); // do not use d_seqEdges directly as it might not be up-to-date
    const SeqTetrahedra& tetrahedra = getTetrahedra(); // do not use d_seqTetrahedra directly as it might not be up-to-date
    m_trianglesInTetrahedron.clear();
    m_trianglesInTetrahedron.resize(tetrahedra.size());

    for (unsigned int i = 0; i < tetrahedra.size(); ++i)
    {
        const Tetra &t=tetrahedra[i];
        // adding triangles in the triangle list of the ith tetrahedron  i
        for (unsigned int j=0; j<4; ++j)
        {
            const TriangleID triangleIndex=getTriangleIndex(t[(j+1)%4],t[(j+2)%4],t[(j+3)%4]);
            assert(triangleIndex != InvalidID);
            m_trianglesInTetrahedron[i][j]=triangleIndex;
        }
    }
}


void MeshTopology::createQuadsAroundVertexArray ()
{
    const SeqQuads& quads = getQuads(); // do not use d_seqQuads directly as it might not be up-to-date
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
    const SeqQuads& quads = getQuads(); // do not use d_seqQuads directly as it might not be up-to-date
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
    //const SeqEdges& edges = getEdges(); // do not use d_seqEdges directly as it might not be up-to-date
    const SeqHexahedra& hexahedra = getHexahedra(); // do not use d_seqHexahedra directly as it might not be up-to-date
    m_quadsInHexahedron.clear();
    m_quadsInHexahedron.resize(hexahedra.size());

    for (unsigned int i = 0; i < hexahedra.size(); ++i)
    {
        const Hexa &h=hexahedra[i];
        QuadID quadIndex;
        // adding the 6 quads in the quad list of the ith hexahedron  i
        // Quad 0 :
        quadIndex=getQuadIndex(h[0],h[3],h[2],h[1]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][0]=quadIndex;
        // Quad 1 :
        quadIndex=getQuadIndex(h[4],h[5],h[6],h[7]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][1]=quadIndex;
        // Quad 2 :
        quadIndex=getQuadIndex(h[0],h[1],h[5],h[4]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][2]=quadIndex;
        // Quad 3 :
        quadIndex=getQuadIndex(h[1],h[2],h[6],h[5]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][3]=quadIndex;
        // Quad 4 :
        quadIndex=getQuadIndex(h[2],h[3],h[7],h[6]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][4]=quadIndex;
        // Quad 5 :
        quadIndex=getQuadIndex(h[3],h[0],h[4],h[7]);
        assert(quadIndex!= InvalidID);
        m_quadsInHexahedron[i][5]=quadIndex;
    }
}

void MeshTopology::createTetrahedraAroundVertexArray ()
{
    m_tetrahedraAroundVertex.resize( nbPoints );
    unsigned int j;

    for (unsigned int i = 0; i < d_seqTetrahedra.getValue().size(); ++i)
    {
        for (j=0; j<4; ++j)
            m_tetrahedraAroundVertex[ d_seqTetrahedra.getValue()[i][j]  ].push_back(i );
    }
}

void MeshTopology::createTetrahedraAroundEdgeArray ()
{
    if (!m_edgesInTetrahedron.size())
        createEdgesInTetrahedronArray();
    m_tetrahedraAroundEdge.resize( getNbEdges() );
    const vector< EdgesInTetrahedron > &tea = m_edgesInTetrahedron;
    unsigned int j;

    for (unsigned int i = 0; i < d_seqTetrahedra.getValue().size(); ++i)
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

    for (unsigned int i = 0; i < d_seqTetrahedra.getValue().size(); ++i)
    {
        for (j=0; j<4; ++j)
            m_tetrahedraAroundTriangle[ tta[i][j] ].push_back( i );
    }
}

void MeshTopology::createHexahedraAroundVertexArray ()
{
    m_hexahedraAroundVertex.resize( nbPoints );
    unsigned int j;

    for (unsigned int i = 0; i < d_seqHexahedra.getValue().size(); ++i)
    {
        for (j=0; j<8; ++j)
            m_hexahedraAroundVertex[ d_seqHexahedra.getValue()[i][j]  ].push_back(i );
    }
}

void MeshTopology::createHexahedraAroundEdgeArray ()
{
    if (!m_edgesInHexahedron.size())
        createEdgesInHexahedronArray();
    m_hexahedraAroundEdge.resize(getNbEdges());
    unsigned int j;
    const vector< EdgesInHexahedron > &hea=m_edgesInHexahedron;

    for (unsigned int i = 0; i < d_seqHexahedra.getValue().size(); ++i)
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

    for (unsigned int i = 0; i < d_seqHexahedra.getValue().size(); ++i)
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

    if (i < m_edgesAroundVertex.size())
        return m_edgesAroundVertex[i];

    return InvalidSet;
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

    if (i <  m_orientedEdgesAroundVertex.size())
        return m_orientedEdgesAroundVertex[i];

    return InvalidSet;
}


const MeshTopology::EdgesInTriangle& MeshTopology::getEdgesInTriangle(TriangleID i)
{
    if (m_edgesInTriangle.empty() || i > m_edgesInTriangle.size()-1)
        createEdgesInTriangleArray();

    if (i < m_edgesInTriangle.size())
        return m_edgesInTriangle[i];

    return InvalidEdgesInTriangles;
}

const MeshTopology::EdgesInQuad& MeshTopology::getEdgesInQuad(QuadID i)
{
    if (m_edgesInQuad.empty() || i > m_edgesInQuad.size()-1)
        createEdgesInQuadArray();

    if (i < m_edgesInQuad.size())
        return m_edgesInQuad[i];

    return InvalidEdgesInQuad;
}

const MeshTopology::EdgesInTetrahedron& MeshTopology::getEdgesInTetrahedron(TetraID i)
{
    if (m_edgesInTetrahedron.empty() || i > m_edgesInTetrahedron.size()-1)
        createEdgesInTetrahedronArray();

    if (i < m_edgesInTetrahedron.size())
        return m_edgesInTetrahedron[i];

    return InvalidEdgesInTetrahedron;
}

const MeshTopology::EdgesInHexahedron& MeshTopology::getEdgesInHexahedron(HexaID i)
{
    if (!m_edgesInHexahedron.size() || i > m_edgesInHexahedron.size()-1)
        createEdgesInHexahedronArray();

    if (i < m_edgesInHexahedron.size())
        return m_edgesInHexahedron[i];

    return InvalidEdgesInHexahedron;
}

const MeshTopology::TrianglesAroundVertex& MeshTopology::getTrianglesAroundVertex(PointID i)
{
    if (!m_trianglesAroundVertex.size() || i > m_trianglesAroundVertex.size()-1)
        createTrianglesAroundVertexArray();

    if (i < m_trianglesAroundVertex.size())
        return m_trianglesAroundVertex[i];

    return InvalidSet;
}

const MeshTopology::TrianglesAroundVertex& MeshTopology::getOrientedTrianglesAroundVertex(PointID i)
{
    if (!m_orientedTrianglesAroundVertex.size() || i > m_orientedTrianglesAroundVertex.size()-1)
        createOrientedTrianglesAroundVertexArray();

    if (i < m_orientedTrianglesAroundVertex.size())
        return m_orientedTrianglesAroundVertex[i];

    return InvalidSet;
}

const MeshTopology::TrianglesAroundEdge& MeshTopology::getTrianglesAroundEdge(EdgeID i)
{
    if (m_trianglesAroundEdge.empty() || i > m_trianglesAroundEdge.size()-1)
        createTrianglesAroundEdgeArray();

    if (i < m_trianglesAroundEdge.size())
        return m_trianglesAroundEdge[i];

    return InvalidSet;
}

const MeshTopology::TrianglesInTetrahedron& MeshTopology::getTrianglesInTetrahedron(TetraID i)
{
    if (!m_trianglesInTetrahedron.size() || i > m_trianglesInTetrahedron.size()-1)
        createTrianglesInTetrahedronArray();

    if (i < m_trianglesInTetrahedron.size())
        return m_trianglesInTetrahedron[i];

    return InvalidTrianglesInTetrahedron;
}

const MeshTopology::QuadsAroundVertex& MeshTopology::getQuadsAroundVertex(PointID i)
{
    if (m_quadsAroundVertex.empty() || i > m_quadsAroundVertex.size()-1)
        createQuadsAroundVertexArray();

    if (i < m_quadsAroundVertex.size())
        return m_quadsAroundVertex[i];

    return InvalidSet;
}

const MeshTopology::QuadsAroundVertex& MeshTopology::getOrientedQuadsAroundVertex(PointID i)
{
    if (m_orientedQuadsAroundVertex.empty() || i > m_orientedQuadsAroundVertex.size()-1)
        createOrientedQuadsAroundVertexArray();

    if (i < m_orientedQuadsAroundVertex.size())
        return m_orientedQuadsAroundVertex[i];

    return InvalidSet;
}

const vector< MeshTopology::QuadID >& MeshTopology::getQuadsAroundEdge(EdgeID i)
{
    if (!m_quadsAroundEdge.size() || i > m_quadsAroundEdge.size()-1)
        createQuadsAroundEdgeArray();


    if(i < m_quadsAroundEdge.size())
        return m_quadsAroundEdge[i];

    return InvalidSet;
}

const MeshTopology::QuadsInHexahedron& MeshTopology::getQuadsInHexahedron(HexaID i)
{
    if (!m_quadsInHexahedron.size() || i > m_quadsInHexahedron.size()-1)
        createQuadsInHexahedronArray();

    if (i < m_quadsInHexahedron.size())
        return m_quadsInHexahedron[i];

    return InvalidQuadsInHexahedron;
}

const MeshTopology::TetrahedraAroundVertex& MeshTopology::getTetrahedraAroundVertex(PointID i)
{
    if (!m_tetrahedraAroundVertex.size() || i > m_tetrahedraAroundVertex.size()-1)
        createTetrahedraAroundVertexArray();

    if (i < m_tetrahedraAroundVertex.size())
        return m_tetrahedraAroundVertex[i];

    return InvalidSet;
}

const MeshTopology::TetrahedraAroundEdge& MeshTopology::getTetrahedraAroundEdge(EdgeID i)
{
    if (!m_tetrahedraAroundEdge.size() || i > m_tetrahedraAroundEdge.size()-1)
        createTetrahedraAroundEdgeArray();

    if (i < m_tetrahedraAroundEdge.size())
        return m_tetrahedraAroundEdge[i];

    return InvalidSet;
}

const MeshTopology::TetrahedraAroundTriangle& MeshTopology::getTetrahedraAroundTriangle(TriangleID i)
{
    if (!m_tetrahedraAroundTriangle.size() || i > m_tetrahedraAroundTriangle.size()-1)
        createTetrahedraAroundTriangleArray();

    if (i < m_tetrahedraAroundTriangle.size())
        return m_tetrahedraAroundTriangle[i];

    return InvalidSet;
}

const MeshTopology::HexahedraAroundVertex& MeshTopology::getHexahedraAroundVertex(PointID i)
{
    if (!m_hexahedraAroundVertex.size() || i > m_hexahedraAroundVertex.size()-1)
        createHexahedraAroundVertexArray();

    if (i < m_hexahedraAroundVertex.size())
        return m_hexahedraAroundVertex[i];

    return InvalidSet;
}

const MeshTopology::HexahedraAroundEdge& MeshTopology::getHexahedraAroundEdge(EdgeID i)
{
    if (!m_hexahedraAroundEdge.size() || i > m_hexahedraAroundEdge.size()-1)
        createHexahedraAroundEdgeArray();

    if (i < m_hexahedraAroundEdge.size())
        return m_hexahedraAroundEdge[i];

    return InvalidSet;
}

const MeshTopology::HexahedraAroundQuad& MeshTopology::getHexahedraAroundQuad(QuadID i)
{
    if (!m_hexahedraAroundQuad.size() || i > m_hexahedraAroundQuad.size()-1)
        createHexahedraAroundQuadArray();

    if (i < m_hexahedraAroundQuad.size())
        return m_hexahedraAroundQuad[i];

    return InvalidSet;
}



const vector< MeshTopology::EdgesAroundVertex >& MeshTopology::getEdgesAroundVertexArray()
{
    if (m_edgesAroundVertex.empty())	// this method should only be called when the array exists.
    {
        dmsg_warning() << "GetEdgesAroundVertexArray: EdgesAroundVertex array is empty. Be sure to call createEdgesAroundVertexArray first.";
        createEdgesAroundVertexArray();
    }

    return m_edgesAroundVertex;
}

const vector< MeshTopology::EdgesInTriangle >& MeshTopology::getEdgesInTriangleArray()
{
    if(m_edgesInTriangle.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetEdgesInTriangleArray: EdgesInTriangle array is empty. Be sure to call createEdgesInTriangleArray first.";
        createEdgesInTriangleArray();
    }

    return m_edgesInTriangle;
}

const vector< MeshTopology::TrianglesAroundVertex >& MeshTopology::getTrianglesAroundVertexArray()
{
    if(m_trianglesAroundVertex.empty())	// this method should only be called when the array exists.
    {
        dmsg_warning() << "GetTrianglesAroundVertexArray: TrianglesAroundVertex array is empty. Be sure to call createTrianglesAroundVertexArray first.";
        createTrianglesAroundVertexArray();
    }

    return m_trianglesAroundVertex;
}

const vector< MeshTopology::TrianglesAroundEdge >& MeshTopology::getTrianglesAroundEdgeArray()
{
    if(m_trianglesAroundEdge.empty())	// this method should only be called when the array exists.
    {
        dmsg_warning() << "GetTrianglesAroundEdgeArray: TrianglesAroundEdge array is empty. Be sure to call createTrianglesAroundEdgeArray first.";
        createTrianglesAroundEdgeArray();
    }

    return m_trianglesAroundEdge;
}




const vector< MeshTopology::EdgesInQuad >& MeshTopology::getEdgesInQuadArray()
{
    if(m_edgesInQuad.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetEdgesInQuadArray: EdgesInQuad array is empty. Be sure to call createEdgesInQuadArray first.";
        createEdgesInQuadArray();
    }

    return m_edgesInQuad;
}

const vector< MeshTopology::QuadsAroundVertex >& MeshTopology::getQuadsAroundVertexArray()
{
    if(m_quadsAroundVertex.empty())	// this method should only be called when the array exists.
    {
        dmsg_warning() << "GetQuadsAroundVertexArray: QuadsAroundVertex array is empty. Be sure to call createQuadsAroundVertexArray first.";
        createQuadsAroundVertexArray();
    }

    return m_quadsAroundVertex;
}

const vector< MeshTopology::QuadsAroundEdge >& MeshTopology::getQuadsAroundEdgeArray()
{
    if(m_quadsAroundEdge.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetQuadsAroundEdgeArray: QuadsAroundEdge array is empty. Be sure to call createQuadsAroundEdgeArray first.";
        createQuadsAroundEdgeArray();
    }

    return m_quadsAroundEdge;
}





const vector< MeshTopology::EdgesInTetrahedron >& MeshTopology::getEdgesInTetrahedronArray()
{
    if (m_edgesInTetrahedron.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetEdgesInTetrahedronArray: EdgesInTetrahedron array is empty. Be sure to call createEdgesInTetrahedronArray first.";
        createEdgesInTetrahedronArray();
    }

    return m_edgesInTetrahedron;
}

const vector< MeshTopology::TrianglesInTetrahedron >& MeshTopology::getTrianglesInTetrahedronArray()
{
    if (m_trianglesInTetrahedron.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetTrianglesInTetrahedronArray: TrianglesInTetrahedron array is empty. Be sure to call createTrianglesInTetrahedronArray first.";
        createTrianglesInTetrahedronArray();
    }

    return m_trianglesInTetrahedron;
}

const vector< MeshTopology::TetrahedraAroundVertex >& MeshTopology::getTetrahedraAroundVertexArray()
{
    if (m_tetrahedraAroundVertex.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetTetrahedraAroundVertexArray: TetrahedraAroundVertex array is empty. Be sure to call createTetrahedraAroundVertexArray first.";
        createTetrahedraAroundVertexArray();
    }

    return m_tetrahedraAroundVertex;
}

const vector< MeshTopology::TetrahedraAroundEdge >& MeshTopology::getTetrahedraAroundEdgeArray()
{
    if (m_tetrahedraAroundEdge.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetTetrahedraAroundEdgeArray: TetrahedraAroundEdge array is empty. Be sure to call createTetrahedraAroundEdgeArray first.";
        createTetrahedraAroundEdgeArray();
    }
    return m_tetrahedraAroundEdge;
}

const vector< MeshTopology::TetrahedraAroundTriangle >& MeshTopology::getTetrahedraAroundTriangleArray()
{
    if (m_tetrahedraAroundTriangle.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetTetrahedraAroundTriangleArray: TetrahedraAroundTriangle array is empty. Be sure to call createTetrahedraAroundTriangleArray first.";
        createTetrahedraAroundTriangleArray();
    }

    return m_tetrahedraAroundTriangle;
}




const vector< MeshTopology::EdgesInHexahedron >& MeshTopology::getEdgesInHexahedronArray()
{
    if (m_edgesInHexahedron.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetEdgesInHexahedronArray: EdgesInHexahedron array is empty. Be sure to call createEdgesInHexahedronArray first.";
        createEdgesInHexahedronArray();
    }

    return m_edgesInHexahedron;
}

const vector< MeshTopology::QuadsInHexahedron >& MeshTopology::getQuadsInHexahedronArray()
{
    if (m_quadsInHexahedron.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetQuadsInHexahedronArray: QuadsInHexahedron array is empty. Be sure to call createQuadsInHexahedronArray first.";
        createQuadsInHexahedronArray();
    }

    return m_quadsInHexahedron;
}

const vector< MeshTopology::HexahedraAroundVertex >& MeshTopology::getHexahedraAroundVertexArray()
{
    if (m_hexahedraAroundVertex.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetHexahedraAroundVertexArray: HexahedraAroundVertex array is empty. Be sure to call createHexahedraAroundVertexArray first.";
        createHexahedraAroundVertexArray();
    }

    return m_hexahedraAroundVertex;
}

const vector< MeshTopology::HexahedraAroundEdge >& MeshTopology::getHexahedraAroundEdgeArray()
{
    if (m_hexahedraAroundEdge.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetHexahedraAroundEdgeArray: HexahedraAroundEdge array is empty. Be sure to call createHexahedraAroundEdgeArray first.";
        createHexahedraAroundEdgeArray();
    }

    return m_hexahedraAroundEdge;
}

const vector< MeshTopology::HexahedraAroundQuad >& MeshTopology::getHexahedraAroundQuadArray()
{
    if (m_hexahedraAroundQuad.empty()) // this method should only be called when the array exists.
    {
        dmsg_warning() << "GetHexahedraAroundQuadArray: HexahedraAroundQuad array is empty. Be sure to call createHexahedraAroundQuadArray first.";
        createHexahedraAroundQuadArray();
    }

    return m_hexahedraAroundQuad;
}



core::topology::Topology::EdgeID MeshTopology::getEdgeIndex(PointID v1, PointID v2)
{
    const EdgesAroundVertex &es1 = getEdgesAroundVertex(v1) ;
    const SeqEdges &ea = getEdges();
    unsigned int i=0;
    EdgeID result= InvalidID;
    while ((i<es1.size()) && (result == InvalidID))
    {
        const MeshTopology::Edge &e=ea[es1[i]];
        if ((e[0]==v2)|| (e[1]==v2))
            result=(int) es1[i];

        i++;
    }

    return result;
}

core::topology::Topology::TriangleID MeshTopology::getTriangleIndex(PointID v1, PointID v2, PointID v3)
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

    if(out2.size() > 1)
        msg_warning() << "More than one triangle found for indices: [" << v1 << "; " << v2 << "; " << v3 << "]";

    if (out2.size()==1)
        return (int) (out2[0]);

    return InvalidID;
}

core::topology::Topology::QuadID MeshTopology::getQuadIndex(PointID v1, PointID v2, PointID v3,  PointID v4)
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

    if(out3.size() > 1)
        msg_warning() << "More than one Quad found for indices: [" << v1 << "; " << v2 << "; " << v3 << "; " << v4 << "]";


    if (out3.size()==1)
        return (int) (out3[0]);

    return InvalidID;
}

core::topology::Topology::TetrahedronID MeshTopology::getTetrahedronIndex(PointID v1, PointID v2, PointID v3,  PointID v4)
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

    if(out3.size() > 1)
        msg_warning() << "More than one Tetrahedron found for indices: [" << v1 << "; " << v2 << "; " << v3 << "; " << v4 << "]";

    if (out3.size()==1)
        return (int) (out3[0]);

    return InvalidID;
}

core::topology::Topology::HexahedronID MeshTopology::getHexahedronIndex(PointID v1, PointID v2, PointID v3, PointID v4, PointID v5, PointID v6, PointID v7, PointID v8)
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
    if(out7.size() > 1)
        msg_warning() << "More than one Hexahedron found for indices: [" << v1 << "; " << v2 << "; " << v3 << "; " << v4 << "; "
                         << v5 << "; " << v6 << "; " << v7 << "; " << v8 << "]";

    if (out7.size()==1)
        return (int) (out7[0]);

    return InvalidID;
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

MeshTopology::Edge MeshTopology::getLocalEdgesInTetrahedron (const HexahedronID i) const
{
    assert(i<6);
    return MeshTopology::Edge (edgesInTetrahedronArray[i][0], edgesInTetrahedronArray[i][1]);
}

MeshTopology::Edge MeshTopology::getLocalEdgesInHexahedron (const HexahedronID i) const
{
    assert(i<12);
    return MeshTopology::Edge (edgesInHexahedronArray[i][0], edgesInHexahedronArray[i][1]);
}


int MeshTopology::computeRelativeOrientationInTri(const PointID ind_p0, const PointID ind_p1, const PointID ind_t)
{
    const Triangle& t = getTriangles()[ind_t];
    Size i = 0;
    while(i < t.size())
    {
        if(ind_p0 == t[i])
            break;
        ++i;
    }

    if(i == t.size()) //ind_p0 is not a PointID in the triangle ind_t
        return 0;

    if(ind_p1 == t[(i+1)%3]) //p0p1 has the same direction of t
        return 1;
    if(ind_p1 == t[(i+2)%3]) //p0p1 has the opposite direction of t
        return -1;

    return 0;
}

int MeshTopology::computeRelativeOrientationInQuad(const PointID ind_p0, const PointID ind_p1, const PointID ind_q)
{
    const Quad& q = getQuads()[ind_q];
    Size i = 0;
    while(i < q.size())
    {
        if(ind_p0 == q[i])
            break;
        ++i;
    }

    if(i == q.size()) //ind_p0 is not a PointID in the quad ind_q
        return 0;

    if(ind_p1 == q[(i+1)%4]) //p0p1 has the same direction of q
        return 1;
    if(ind_p1 == q[(i+3)%4]) //p0p1 has the opposite direction of q
        return -1;

    return 0;
}

void MeshTopology::reOrientateTriangle(TriangleID id)
{
    if (id >= this->getNbTriangles())
    {
        msg_warning() << "reOrientateTriangle Triangle ID out of bounds.";
        return;
    }
    Triangle& tri = (*d_seqTriangles.beginEdit())[id];
    const unsigned int tmp = tri[1];
    tri[1] = tri[2];
    tri[2] = tmp;
    d_seqTriangles.endEdit();

    return;
}

bool MeshTopology::hasPos() const
{
    return !d_seqPoints.getValue().empty();
}

SReal MeshTopology::getPX(Index i) const
{
    const auto& points = d_seqPoints.getValue();
    return ((unsigned)i<points.size()?points[i][0]:0.0);
}

SReal MeshTopology::getPY(Index i) const
{
    const auto& points = d_seqPoints.getValue();
    return ((unsigned)i<points.size()?points[i][1]:0.0);
}

SReal MeshTopology::getPZ(Index i) const
{
    const auto& points = d_seqPoints.getValue();
    return ((unsigned)i<points.size()?points[i][2]:0.0);
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
    //msg_info() << "MeshTopology::invalidate()";
}



void MeshTopology::updateHexahedra()
{
    if (!d_seqHexahedra.getValue().empty()) return; // hexahedra already defined
    // No 4D elements yet! ;)
}

void MeshTopology::updateTetrahedra()
{
    if (!d_seqTetrahedra.getValue().empty()) return; // tetrahedra already defined
    // No 4D elements yet! ;)
}




/// Get information about connexity of the mesh
/// @{

bool MeshTopology::checkConnexity()
{
    Size nbr = 0;

    if (m_upperElementType == geometry::ElementType::HEXAHEDRON)
        nbr = this->getNbHexahedra();
    else if (m_upperElementType == geometry::ElementType::TETRAHEDRON)
        nbr = this->getNbTetrahedra();
    else if (m_upperElementType == geometry::ElementType::QUAD)
        nbr = this->getNbQuads();
    else if (m_upperElementType == geometry::ElementType::TRIANGLE)
        nbr = this->getNbTriangles();
    else
        nbr = this->getNbEdges();

    if (nbr == 0)
    {
        msg_error() << "CheckConnexity: Can't compute connexity as some element are missing.";
        return false;
    }

    const auto elemAll = this->getConnectedElement(0);

    if (elemAll.size() != nbr)
    {
        msg_error() << "CheckConnexity: elements are missing. There is more than one connexe component.";
        return false;
    }

    return true;
}


Size MeshTopology::getNumberOfConnectedComponent()
{
    Size nbr = 0;

    if (m_upperElementType == geometry::ElementType::HEXAHEDRON)
        nbr = this->getNbHexahedra();
    else if (m_upperElementType == geometry::ElementType::TETRAHEDRON)
        nbr = this->getNbTetrahedra();
    else if (m_upperElementType == geometry::ElementType::QUAD)
        nbr = this->getNbQuads();
    else if (m_upperElementType == geometry::ElementType::TRIANGLE)
        nbr = this->getNbTriangles();
    else
        nbr = this->getNbEdges();

    if (nbr == 0)
    {
        msg_error() << "GetNumberOfConnectedComponent Can't compute connexity as there are no elements.";
        return 0;
    }

    auto elemAll = this->getConnectedElement(0);
    unsigned int cpt = 1;

    while (elemAll.size() < nbr)
    {
        std::sort(elemAll.begin(), elemAll.end());
        Index other_ID = Index(elemAll.size());

        for (Size i = 0; i<elemAll.size(); ++i)
            if (elemAll[i] != i)
            {
                other_ID = i;
                break;
            }

        auto elemTmp = this->getConnectedElement(other_ID);
        cpt++;

        elemAll.insert(elemAll.begin(), elemTmp.begin(), elemTmp.end());
    }

    return cpt;
}


const sofa::type::vector<Index> MeshTopology::getConnectedElement(Index elem)
{
    Size nbr = 0;

    if (m_upperElementType == geometry::ElementType::HEXAHEDRON)
        nbr = this->getNbHexahedra();
    else if (m_upperElementType == geometry::ElementType::TETRAHEDRON)
        nbr = this->getNbTetrahedra();
    else if (m_upperElementType == geometry::ElementType::QUAD)
        nbr = this->getNbQuads();
    else if (m_upperElementType == geometry::ElementType::TRIANGLE)
        nbr = this->getNbTriangles();
    else
        nbr = this->getNbEdges();

    sofa::type::vector<Index> elemAll;
    sofa::type::vector<Index> elemOnFront, elemPreviousFront, elemNextFront;
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
        for (unsigned int id : elemNextFront)
        {
            bool find = false;
            for (const unsigned int j : elemAll)
            {
                if (id == j)
                {
                    find = true;
                    break;
                }
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
            msg_error() << "Loop for computing connexity has reach end.";
        }

        // iterate
        elemOnFront = elemPreviousFront;
        elemPreviousFront.clear();
    }

    return elemAll;
}


const sofa::type::vector<Index> MeshTopology::getElementAroundElement(Index elem)
{
    sofa::type::vector<Index> elems;
    unsigned int nbr = 0;

    if (m_upperElementType == geometry::ElementType::HEXAHEDRON)
    {
        nbr = 8;
        if(!this->m_hexahedraAroundVertex.empty())
            createHexahedraAroundVertexArray();
    }
    else if (m_upperElementType == geometry::ElementType::TETRAHEDRON)
    {
        nbr = 4;
        if(!this->m_tetrahedraAroundVertex.empty())
            createTetrahedraAroundVertexArray();
    }
    else if (m_upperElementType == geometry::ElementType::QUAD)
    {
        nbr = 4;
        if(!this->m_quadsAroundVertex.empty())
            createQuadsAroundVertexArray();
    }
    else if (m_upperElementType == geometry::ElementType::TRIANGLE)
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
        sofa::type::vector<Index> elemAV;

        if (m_upperElementType == geometry::ElementType::HEXAHEDRON)
            elemAV = this->getHexahedraAroundVertex(getHexahedron(elem)[i]);
        else if (m_upperElementType == geometry::ElementType::TETRAHEDRON)
            elemAV = this->getTetrahedraAroundVertex(getTetrahedron(elem)[i]);
        else if (m_upperElementType == geometry::ElementType::QUAD)
            elemAV = this->getQuadsAroundVertex(getQuad(elem)[i]);
        else if (m_upperElementType == geometry::ElementType::TRIANGLE)
            elemAV = this->getTrianglesAroundVertex(getTriangle(elem)[i]);
        else
            elemAV = this->getEdgesAroundVertex(getEdge(elem)[i]);


        for (const unsigned int id : elemAV) // for each element around the node
        {
            bool find = false;
            if (id == elem)
                continue;

            for (const unsigned int elem : elems) // check no redundancy
                if (id == elem)
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


const sofa::type::vector<Index> MeshTopology::getElementAroundElements(sofa::type::vector<Index> elems)
{
    sofa::type::vector<Index> elemAll;
    sofa::type::vector<Index> elemTmp;

    if (m_upperElementType == geometry::ElementType::HEXAHEDRON)
    {
        if(!this->m_hexahedraAroundVertex.empty())
            createHexahedraAroundVertexArray();
    }
    else if (m_upperElementType == geometry::ElementType::TETRAHEDRON)
    {
        if(!this->m_tetrahedraAroundVertex.empty())
            createTetrahedraAroundVertexArray();
    }
    else if (m_upperElementType == geometry::ElementType::QUAD)
    {
        if(!this->m_quadsAroundVertex.empty())
            createQuadsAroundVertexArray();
    }
    else if (m_upperElementType == geometry::ElementType::TRIANGLE)
    {
        if(!this->m_trianglesAroundVertex.empty())
            createTrianglesAroundVertexArray();
    }
    else
    {
        if(!this->m_edgesAroundVertex.empty())
            createEdgesAroundVertexArray();
    }


    for (const unsigned int elem : elems) // for each elementID of input vector
    {
        sofa::type::vector<Index> elemTmp2 = this->getElementAroundElement(elem);

        elemTmp.insert(elemTmp.end(), elemTmp2.begin(), elemTmp2.end());
    }

    for (const unsigned int id : elemTmp) // for each elementID found
    {
        bool find = false;
        for (const unsigned int elem : elems) // check no redundancy with input vector
            if (id == elem)
            {
                find = true;
                break;
            }

        if (!find)
        {
            for (const unsigned int j : elemAll) // check no redundancy in output vector
                if (id == j)
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

SReal MeshTopology::getPosX(Index i) const
{
    return ((unsigned)i < d_seqPoints.getValue().size() ? d_seqPoints.getValue()[i][0] : 0.0);
}

SReal MeshTopology::getPosY(Index i) const
{
    return ((unsigned)i < d_seqPoints.getValue().size() ? d_seqPoints.getValue()[i][1] : 0.0);
}

SReal MeshTopology::getPosZ(Index i) const
{
    return ((unsigned)i < d_seqPoints.getValue().size() ? d_seqPoints.getValue()[i][2] : 0.0);
}

void MeshTopology::draw(const core::visual::VisualParams* vparams)
{
    const auto stateLifeCycle = vparams->drawTool()->makeStateLifeCycle();

    // Draw Edges
    if(d_drawEdges.getValue())
    {
        std::vector<type::Vec3> pos;
        pos.reserve(this->getNbEdges()*2u);
        for (EdgeID i=0; i<getNbEdges(); i++)
        {
            const Edge& c = getEdge(i);
            pos.emplace_back(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0]));
            pos.emplace_back(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1]));
        }
        vparams->drawTool()->drawLines(pos, 1.0f, sofa::type::RGBAColor(0.4f,1.0f,0.3f,1.0f));
    }

    //Draw Triangles
    if(d_drawTriangles.getValue())
    {
        std::vector<type::Vec3> pos;
        pos.reserve(this->getNbTriangles()*3u);
        for (TriangleID i=0; i<getNbTriangles(); i++)
        {
            const Triangle& c = getTriangle(i);
            pos.emplace_back(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0]));
            pos.emplace_back(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1]));
            pos.emplace_back(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2]));
        }
        vparams->drawTool()->drawTriangles(pos, sofa::type::RGBAColor(0.4f,1.0f,0.3f,1.0f));
    }

    //Draw Quads
    if(d_drawQuads.getValue())
    {
        std::vector<type::Vec3> pos;
        pos.reserve(this->getNbQuads()*4u);
        for (QuadID i=0; i<getNbQuads(); i++)
        {
            const Quad& c = getQuad(i);
            pos.emplace_back(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0]));
            pos.emplace_back(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1]));
            pos.emplace_back(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2]));
            pos.emplace_back(getPosX(c[3]), getPosY(c[3]), getPosZ(c[3]));
        }
        vparams->drawTool()->drawQuads(pos, sofa::type::RGBAColor(0.4f,1.0f,0.3f,1.0f));
    }

    //Draw Hexahedron
    if (d_drawHexa.getValue())
    {
        std::vector<type::Vec3> pos1;
        std::vector<type::Vec3> pos2;
        pos1.reserve(this->getNbHexahedra()*8u);
        pos2.reserve(this->getNbHexahedra()*8u);
        for (HexahedronID i=0; i<getNbHexahedra(); i++)
        {
            const Hexa& c = getHexahedron(i);
            pos1.emplace_back(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0]));
            pos1.emplace_back(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1]));
            pos1.emplace_back(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2]));
            pos1.emplace_back(getPosX(c[3]), getPosY(c[3]), getPosZ(c[3]));
            pos1.emplace_back(getPosX(c[4]), getPosY(c[4]), getPosZ(c[4]));
            pos1.emplace_back(getPosX(c[5]), getPosY(c[5]), getPosZ(c[5]));
            pos1.emplace_back(getPosX(c[6]), getPosY(c[6]), getPosZ(c[6]));
            pos1.emplace_back(getPosX(c[7]), getPosY(c[7]), getPosZ(c[7]));

            pos2.emplace_back(getPosX(c[3]), getPosY(c[3]), getPosZ(c[3]));
            pos2.emplace_back(getPosX(c[7]), getPosY(c[7]), getPosZ(c[7]));
            pos2.emplace_back(getPosX(c[2]), getPosY(c[2]), getPosZ(c[2]));
            pos2.emplace_back(getPosX(c[6]), getPosY(c[6]), getPosZ(c[6]));
            pos2.emplace_back(getPosX(c[0]), getPosY(c[0]), getPosZ(c[0]));
            pos2.emplace_back(getPosX(c[4]), getPosY(c[4]), getPosZ(c[4]));
            pos2.emplace_back(getPosX(c[1]), getPosY(c[1]), getPosZ(c[1]));
            pos2.emplace_back(getPosX(c[5]), getPosY(c[5]), getPosZ(c[5]));
        }
        vparams->drawTool()->drawQuads(pos1, sofa::type::RGBAColor(0.4f,1.0f,0.3f,1.0f));
        vparams->drawTool()->drawLines(pos2, 1.0f, sofa::type::RGBAColor(0.4f,1.0f,0.3f,1.0f));
    }

    // Draw Tetra
    if(d_drawTetra.getValue())
    {
        std::vector<type::Vec3> pos;
        pos.reserve(this->getNbTetrahedra()*12u);
        for (TetrahedronID i=0; i<getNbTetras(); i++)
        {
            const Tetra& t = getTetra(i);
            pos.emplace_back(getPosX(t[0]), getPosY(t[0]), getPosZ(t[0]));
            pos.emplace_back(getPosX(t[1]), getPosY(t[1]), getPosZ(t[1]));

            pos.emplace_back(getPosX(t[0]), getPosY(t[0]), getPosZ(t[0]));
            pos.emplace_back(getPosX(t[2]), getPosY(t[2]), getPosZ(t[2]));

            pos.emplace_back(getPosX(t[0]), getPosY(t[0]), getPosZ(t[0]));
            pos.emplace_back(getPosX(t[3]), getPosY(t[3]), getPosZ(t[3]));

            pos.emplace_back(getPosX(t[1]), getPosY(t[1]), getPosZ(t[1]));
            pos.emplace_back(getPosX(t[2]), getPosY(t[2]), getPosZ(t[2]));

            pos.emplace_back(getPosX(t[1]), getPosY(t[1]), getPosZ(t[1]));
            pos.emplace_back(getPosX(t[3]), getPosY(t[3]), getPosZ(t[3]));

            pos.emplace_back(getPosX(t[2]), getPosY(t[2]), getPosZ(t[2]));
            pos.emplace_back(getPosX(t[3]), getPosY(t[3]), getPosZ(t[3]));
        }
        vparams->drawTool()->drawLines(pos, 1.0f, sofa::type::RGBAColor(1.0f,0.0f,0.0f,1.0f));
    }


}

} //namespace sofa::component::topology::container::constant
