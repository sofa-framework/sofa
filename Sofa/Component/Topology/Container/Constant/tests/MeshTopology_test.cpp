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

#include <sofa/component/topology/testing/fake_TopologyScene.h>
#include <sofa/testing/BaseTest.h>

#include <sofa/component/topology/container/constant/MeshTopology.h>
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>


#include <sofa/helper/system/FileRepository.h>

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::component::topology::container::constant;
using namespace sofa::core::topology;
using namespace sofa::testing;

/**
 * This class will test the MeshTopology containers.
 * All cross buffer tests are already done in the Edge/Triangle/..Topology_test classes
 * Thus only access to buffers are tested and the content is compare to dynamyc container.
*/
class MeshTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();

    bool testHexahedronTopology();
    bool testTetrahedronTopology();
    
    bool testQuadTopology();
    bool testTriangleTopology();

    bool testEdgeTopology();
    bool testVertexTopology();
     
};


bool MeshTopology_test::testEmptyContainer()
{
    const MeshTopology::SPtr topoCon = sofa::core::objectmodel::New< MeshTopology >();

    EXPECT_EQ(topoCon->getNbHexahedra(), 0);
    EXPECT_EQ(topoCon->getHexahedra().size(), 0);
    
    EXPECT_EQ(topoCon->getNbHexahedra(), 0);
    EXPECT_EQ(topoCon->getTetrahedra().size(), 0);

    EXPECT_EQ(topoCon->getNbQuads(), 0);
    EXPECT_EQ(topoCon->getQuads().size(), 0);

    EXPECT_EQ(topoCon->getNbTriangles(), 0);
    EXPECT_EQ(topoCon->getTriangles().size(), 0);

    EXPECT_EQ(topoCon->getNbEdges(), 0);
    EXPECT_EQ(topoCon->getEdges().size(), 0);

    EXPECT_EQ(topoCon->getNbPoints(), 0);

    return true;
}


bool MeshTopology_test::testHexahedronTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON);
    HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    fake_TopologyScene* scene2 = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON, true);
    MeshTopology* topo = dynamic_cast<MeshTopology*>(scene2->getNode().get()->getMeshTopology());
    topo->init();
    
    if (topoCon == nullptr || topo == nullptr)
    {
        if (scene != nullptr)
            delete scene;

        if (scene2 != nullptr)
            delete scene2;

        return false;
    }

    int nbrHexahedron = 9;
    int elemSize = 8;

    // Check tetrahedra container buffers size
    EXPECT_EQ(topo->getNbHexahedra(), nbrHexahedron);
    EXPECT_EQ(topo->getHexahedra().size(), nbrHexahedron);

    //// check tetrahedron buffer    
    const sofa::type::vector<HexahedronSetTopologyContainer::Hexahedron>& hexahedra1 = topoCon->getHexahedronArray();
    const BaseMeshTopology::SeqHexahedra& hexahedra2 = topo->getHexahedra();

    // Check hexahedron buffer access    
    const BaseMeshTopology::Hexahedron& hexahedron1 = topo->getHexahedron(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(hexahedron1[i], hexahedra2[1][i]);

    // check buffer
    EXPECT_EQ(hexahedra1.size(), hexahedra2.size());
    for (int i = 0; i < nbrHexahedron; ++i) // test the 9.
        for (int j = 0; j < elemSize; ++j)
            EXPECT_EQ(hexahedra1[i][j], hexahedra2[i][j]);


    //// create and get cross elements buffers
    const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundQuad >& hexahedraAroundQuad1 = topoCon->getHexahedraAroundQuadArray();
    const sofa::type::vector< HexahedronSetTopologyContainer::QuadsInHexahedron > & trianglesInHexahedron1 = topoCon->getQuadsInHexahedronArray();
    const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundEdge >& hexahedraAroundEdge1 = topoCon->getHexahedraAroundEdgeArray();
    const sofa::type::vector< HexahedronSetTopologyContainer::EdgesInHexahedron > & edgesInHexahedron1 = topoCon->getEdgesInHexahedronArray();
    const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundVertex >& hexahedraAroundVertex1 = topoCon->getHexahedraAroundVertexArray();
    const HexahedronSetTopologyContainer::SeqEdges& edges1 = topoCon->getEdges();

    const sofa::type::vector< BaseMeshTopology::HexahedraAroundQuad > &hexahedraAroundQuad2 = topo->getHexahedraAroundQuadArray();
    const sofa::type::vector< BaseMeshTopology::QuadsInHexahedron > &trianglesInHexahedron2 = topo->getQuadsInHexahedronArray();
    const sofa::type::vector< BaseMeshTopology::HexahedraAroundEdge >& hexahedraAroundEdge2 = topo->getHexahedraAroundEdgeArray();
    const sofa::type::vector< BaseMeshTopology::EdgesInHexahedron >& edgesInHexahedron2 = topo->getEdgesInHexahedronArray();
    const sofa::type::vector< BaseMeshTopology::HexahedraAroundVertex >& hexahedraAroundVertex2 = topo->getHexahedraAroundVertexArray();
    const BaseMeshTopology::SeqEdges& edges2 = topo->getEdges();

    // check all buffers size
    EXPECT_EQ(hexahedraAroundQuad1.size(), hexahedraAroundQuad2.size());
    EXPECT_EQ(trianglesInHexahedron1.size(), trianglesInHexahedron2.size());
    EXPECT_EQ(hexahedraAroundEdge1.size(), hexahedraAroundEdge2.size());
    EXPECT_EQ(edgesInHexahedron1.size(), edgesInHexahedron2.size());
    EXPECT_EQ(hexahedraAroundVertex1.size(), hexahedraAroundVertex2.size());
    EXPECT_EQ(edges1.size(), edges2.size());
    EXPECT_EQ(topoCon->getNbPoints(), topo->getNbPoints());

    for (int i = 0; i < 6; ++i) // only test the 6 firsts elements of each buffer
    {
        const HexahedronSetTopologyContainer::HexahedraAroundQuad& HaQ1 = hexahedraAroundQuad1[i];
        const BaseMeshTopology::HexahedraAroundQuad& HaQ2 = hexahedraAroundQuad2[i];
        EXPECT_EQ(HaQ1.size(), HaQ2.size());
        for (size_t j = 0; j<HaQ1.size(); ++j)
            EXPECT_EQ(HaQ1[j], HaQ2[j]);

        const HexahedronSetTopologyContainer::QuadsInHexahedron& QiH1 = trianglesInHexahedron1[i];
        const BaseMeshTopology::QuadsInHexahedron& QiH2 = trianglesInHexahedron2[i];
        EXPECT_EQ(QiH1.size(), QiH2.size());
        for (size_t j = 0; j<QiH1.size(); ++j)
            EXPECT_EQ(QiH1[j], QiH1[j]);

        const HexahedronSetTopologyContainer::HexahedraAroundEdge& HaE1 = hexahedraAroundEdge1[i];
        const BaseMeshTopology::HexahedraAroundEdge& HaE2 = hexahedraAroundEdge2[i];
        EXPECT_EQ(HaE1.size(), HaE2.size());
        for (size_t j = 0; j<HaE1.size(); ++j)
            EXPECT_EQ(HaE1[j], HaE2[j]);

        const HexahedronSetTopologyContainer::EdgesInHexahedron& EiH1 = edgesInHexahedron1[i];
        const BaseMeshTopology::EdgesInHexahedron& EiH2 = edgesInHexahedron2[i];
        EXPECT_EQ(EiH1.size(), EiH2.size());
        for (size_t j = 0; j<EiH1.size(); ++j)
            EXPECT_EQ(EiH1[j], EiH2[j]);

        const HexahedronSetTopologyContainer::HexahedraAroundVertex& HaV1 = hexahedraAroundVertex1[i];
        const BaseMeshTopology::HexahedraAroundVertex& HaV2 = hexahedraAroundVertex2[i];
        EXPECT_EQ(HaV1.size(), HaV2.size());
        for (size_t j = 0; j<HaV1.size(); ++j)
            EXPECT_EQ(HaV1[j], HaV2[j]);

        const HexahedronSetTopologyContainer::Edge& e1 = edges1[i];
        const BaseMeshTopology::Edge& e2 = edges2[i];
        for (int j = 0; j<2; ++j)
            EXPECT_EQ(e1[j], e2[j]);
    }

    if (scene != nullptr)
        delete scene;
    if (scene2 != nullptr)
        delete scene2;

    return true;
}


bool MeshTopology_test::testTetrahedronTopology()
{   
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON);
    TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    fake_TopologyScene* scene2 = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON, true);
    MeshTopology* topo = dynamic_cast<MeshTopology*>(scene2->getNode().get()->getMeshTopology());
    topo->init();

    if (topoCon == nullptr || topo == nullptr)
    {
        if (scene != nullptr)
            delete scene;

        if (scene2 != nullptr)
            delete scene2;

        return false;
    }

    int nbrTetrahedron = 44;
    int elemSize = 4;

    // Check tetrahedra container buffers size
    EXPECT_EQ(topo->getNbTetrahedra(), nbrTetrahedron);
    EXPECT_EQ(topo->getTetrahedra().size(), nbrTetrahedron);

    //// check tetrahedron buffer    
    const sofa::type::vector<TetrahedronSetTopologyContainer::Tetrahedron>& tetrahedra1 = topoCon->getTetrahedronArray();
    const BaseMeshTopology::SeqTetrahedra& tetrahedra2 = topo->getTetrahedra();

    // Check tetrahedron buffer access    
    const BaseMeshTopology::Tetrahedron& tetrahedron1 = topo->getTetrahedron(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(tetrahedron1[i], tetrahedra2[1][i]);

    // check buffer
    EXPECT_EQ(tetrahedra1.size(), tetrahedra2.size());
    for (int i = 0; i < 10; ++i) // only the 10 first elements
        for (int j = 0; j < elemSize; ++j)
            EXPECT_EQ(tetrahedra1[i][j], tetrahedra2[i][j]);


    //// create and get cross elements buffers
    const sofa::type::vector< TetrahedronSetTopologyContainer::TetrahedraAroundTriangle >& tetrahedraAroundTriangle1 = topoCon->getTetrahedraAroundTriangleArray();
    const sofa::type::vector< TetrahedronSetTopologyContainer::TrianglesInTetrahedron > & trianglesInTetrahedron1 = topoCon->getTrianglesInTetrahedronArray();
    const sofa::type::vector< TetrahedronSetTopologyContainer::TetrahedraAroundEdge >& tetrahedraAroundEdge1 = topoCon->getTetrahedraAroundEdgeArray();
    const sofa::type::vector< TetrahedronSetTopologyContainer::EdgesInTetrahedron > & edgesInTetrahedron1 = topoCon->getEdgesInTetrahedronArray();
    const sofa::type::vector< TetrahedronSetTopologyContainer::TetrahedraAroundVertex >& tetrahedraAroundVertex1 = topoCon->getTetrahedraAroundVertexArray();
    const TetrahedronSetTopologyContainer::SeqEdges& edges1 = topoCon->getEdges();

    const sofa::type::vector< BaseMeshTopology::TetrahedraAroundTriangle > &tetrahedraAroundTriangle2 = topo->getTetrahedraAroundTriangleArray();
    const sofa::type::vector< BaseMeshTopology::TrianglesInTetrahedron > &trianglesInTetrahedron2 = topo->getTrianglesInTetrahedronArray();
    const sofa::type::vector< BaseMeshTopology::TetrahedraAroundEdge >& tetrahedraAroundEdge2 = topo->getTetrahedraAroundEdgeArray();
    const sofa::type::vector< BaseMeshTopology::EdgesInTetrahedron >& edgesInTetrahedron2 = topo->getEdgesInTetrahedronArray();
    const sofa::type::vector< BaseMeshTopology::TetrahedraAroundVertex >& tetrahedraAroundVertex2 = topo->getTetrahedraAroundVertexArray();    
    const BaseMeshTopology::SeqEdges& edges2 = topo->getEdges();

    // check all buffers size
    EXPECT_EQ(tetrahedraAroundTriangle1.size(), tetrahedraAroundTriangle2.size());
    EXPECT_EQ(trianglesInTetrahedron1.size(), trianglesInTetrahedron2.size());
    EXPECT_EQ(tetrahedraAroundEdge1.size(), tetrahedraAroundEdge2.size());
    EXPECT_EQ(edgesInTetrahedron1.size(), edgesInTetrahedron2.size());
    EXPECT_EQ(tetrahedraAroundVertex1.size(), tetrahedraAroundVertex2.size());    
    EXPECT_EQ(edges1.size(), edges2.size());
    EXPECT_EQ(topoCon->getNbPoints(), topo->getNbPoints());

    for (int i = 0; i < 6; ++i) // only test the 6 firsts elements of each buffer
    {
        const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle& TaT1 = tetrahedraAroundTriangle1[i];
        const BaseMeshTopology::TetrahedraAroundTriangle& TaT2 = tetrahedraAroundTriangle2[i];
        EXPECT_EQ(TaT1.size(), TaT2.size());
        for (size_t j = 0; j<TaT1.size(); ++j)
            EXPECT_EQ(TaT1[j], TaT2[j]);

        const TetrahedronSetTopologyContainer::TrianglesInTetrahedron& TiT1 = trianglesInTetrahedron1[i];
        const BaseMeshTopology::TrianglesInTetrahedron& TiT2 = trianglesInTetrahedron2[i];
        EXPECT_EQ(TiT1.size(), TiT2.size());
        for (size_t j = 0; j<TiT1.size(); ++j)
            EXPECT_EQ(TiT1[j], TiT1[j]);

        const TetrahedronSetTopologyContainer::TetrahedraAroundEdge& TaE1 = tetrahedraAroundEdge1[i];
        const BaseMeshTopology::TetrahedraAroundEdge& TaE2 = tetrahedraAroundEdge2[i];
        EXPECT_EQ(TaE1.size(), TaE2.size());
        for (size_t j = 0; j<TaE1.size(); ++j)
            EXPECT_EQ(TaE1[j], TaE2[j]);

        const TetrahedronSetTopologyContainer::EdgesInTetrahedron& EiT1 = edgesInTetrahedron1[i];
        const BaseMeshTopology::EdgesInTetrahedron& EiT2 = edgesInTetrahedron2[i];
        EXPECT_EQ(EiT1.size(), EiT2.size());
        for (size_t j = 0; j<EiT1.size(); ++j)
            EXPECT_EQ(EiT1[j], EiT2[j]);

        const TetrahedronSetTopologyContainer::TetrahedraAroundVertex& TaV1 = tetrahedraAroundVertex1[i];
        const BaseMeshTopology::TetrahedraAroundVertex& TaV2 = tetrahedraAroundVertex2[i];
        EXPECT_EQ(TaV1.size(), TaV2.size());
        for (size_t j = 0; j<TaV1.size(); ++j)
            EXPECT_EQ(TaV1[j], TaV2[j]);
        
        const TetrahedronSetTopologyContainer::Edge& e1 = edges1[i];
        const BaseMeshTopology::Edge& e2 = edges2[i];
        for (int j = 0; j<2; ++j)
            EXPECT_EQ(e1[j], e2[j]);
    }

    if (scene != nullptr)
        delete scene;
    if (scene2 != nullptr)
        delete scene2;

    return true;
}


bool MeshTopology_test::testQuadTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_quads.obj", sofa::geometry::ElementType::QUAD);
    QuadSetTopologyContainer* topoCon = dynamic_cast<QuadSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    fake_TopologyScene* scene2 = new fake_TopologyScene("mesh/square1_quads.obj", sofa::geometry::ElementType::QUAD, true);
    MeshTopology* topo = dynamic_cast<MeshTopology*>(scene2->getNode().get()->getMeshTopology());
    topo->init();

    if (topoCon == nullptr || topo == nullptr)
    {
        if (scene != nullptr)
            delete scene;

        if (scene2 != nullptr)
            delete scene2;

        return false;
    }

    int nbrQuad = 9;
    int elemSize = 4;

    // Check quads container buffers size
    EXPECT_EQ(topo->getNbQuads(), nbrQuad);
    EXPECT_EQ(topo->getQuads().size(), nbrQuad);

    //// check quad buffer    
    const sofa::type::vector<QuadSetTopologyContainer::Quad>& quads1 = topoCon->getQuadArray();
    const BaseMeshTopology::SeqQuads& quads2 = topo->getQuads();

    // Check quad buffer access    
    const BaseMeshTopology::Quad& quad1 = topo->getQuad(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(quad1[i], quads2[1][i]);

    // check buffer
    EXPECT_EQ(quads1.size(), quads2.size());
    for (int i = 0; i < nbrQuad; ++i)
        for (int j = 0; j < elemSize; ++j)
            EXPECT_EQ(quads1[i][j], quads2[i][j]);


    //// create and get cross elements buffers
    const sofa::type::vector< QuadSetTopologyContainer::EdgesInQuad > & edgesInQuad1 = topoCon->getEdgesInQuadArray();
    const sofa::type::vector< QuadSetTopologyContainer::QuadsAroundVertex >& quadsAroundVertex1 = topoCon->getQuadsAroundVertexArray();
    const sofa::type::vector< QuadSetTopologyContainer::QuadsAroundEdge >& quadsAroundEdge1 = topoCon->getQuadsAroundEdgeArray();
    const QuadSetTopologyContainer::SeqEdges& edges1 = topoCon->getEdges();

    const sofa::type::vector< BaseMeshTopology::EdgesInQuad >& edgesInQuad2 = topo->getEdgesInQuadArray();
    const sofa::type::vector< BaseMeshTopology::QuadsAroundVertex >& quadsAroundVertex2 = topo->getQuadsAroundVertexArray();
    const sofa::type::vector< BaseMeshTopology::QuadsAroundEdge >& quadsAroundEdge2 = topo->getQuadsAroundEdgeArray();
    const BaseMeshTopology::SeqEdges& edges2 = topo->getEdges();

    // check all buffers size
    EXPECT_EQ(edgesInQuad1.size(), edgesInQuad2.size());
    EXPECT_EQ(quadsAroundVertex1.size(), quadsAroundVertex2.size());
    EXPECT_EQ(quadsAroundEdge1.size(), quadsAroundEdge2.size());
    EXPECT_EQ(edges1.size(), edges2.size());
    EXPECT_EQ(topoCon->getNbPoints(), topo->getNbPoints());

    for (int i = 0; i < 6; ++i) // only test the 6 firsts elements of each buffer
    {
        const QuadSetTopologyContainer::EdgesInQuad& EiQ1 = edgesInQuad1[i];
        const BaseMeshTopology::EdgesInQuad& EiQ2 = edgesInQuad2[i];
        EXPECT_EQ(EiQ1.size(), EiQ2.size());
        for (size_t j = 0; j<EiQ1.size(); ++j)
            EXPECT_EQ(EiQ1[j], EiQ2[j]);

        const QuadSetTopologyContainer::QuadsAroundVertex& QaV1 = quadsAroundVertex1[i];
        const BaseMeshTopology::QuadsAroundVertex& QaV2 = quadsAroundVertex2[i];
        EXPECT_EQ(QaV1.size(), QaV2.size());
        for (size_t j = 0; j<QaV1.size(); ++j)
            EXPECT_EQ(QaV1[j], QaV2[j]);

        const QuadSetTopologyContainer::QuadsAroundEdge& QaE1 = quadsAroundEdge1[i];
        const BaseMeshTopology::QuadsAroundEdge& QaE2 = quadsAroundEdge2[i];
        EXPECT_EQ(QaE1.size(), QaE2.size());
        for (size_t j = 0; j<QaE1.size(); ++j)
            EXPECT_EQ(QaE1[j], QaE2[j]);

        const QuadSetTopologyContainer::Edge& e1 = edges1[i];
        const BaseMeshTopology::Edge& e2 = edges2[i];
        for (int j = 0; j<2; ++j)
            EXPECT_EQ(e1[j], e2[j]);
    }

    if (scene != nullptr)
        delete scene;
    if (scene2 != nullptr)
        delete scene2;

    return true;
}


bool MeshTopology_test::testTriangleTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1.obj", sofa::geometry::ElementType::TRIANGLE);
    TriangleSetTopologyContainer* topoCon = dynamic_cast<TriangleSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    fake_TopologyScene* scene2 = new fake_TopologyScene("mesh/square1.obj", sofa::geometry::ElementType::TRIANGLE, true);
    MeshTopology* topo = dynamic_cast<MeshTopology*>(scene2->getNode().get()->getMeshTopology());
    topo->init();

    if (topoCon == nullptr || topo == nullptr)
    {
        if (scene != nullptr)
            delete scene;

        if (scene2 != nullptr)
            delete scene2;

        return false;
    }

    int nbrTriangle = 26;
    int elemSize = 3;

    // Check triangles container buffers size
    EXPECT_EQ(topo->getNbTriangles(), nbrTriangle);
    EXPECT_EQ(topo->getTriangles().size(), nbrTriangle);

    //// check triangle buffer    
    const sofa::type::vector<TriangleSetTopologyContainer::Triangle>& triangles1 = topoCon->getTriangleArray();
    const BaseMeshTopology::SeqTriangles& triangles2 = topo->getTriangles();

    // Check triangle buffer access    
    const BaseMeshTopology::Triangle& triangle1 = topo->getTriangle(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(triangle1[i], triangles2[1][i]);

    // check buffer
    EXPECT_EQ(triangles1.size(), triangles2.size());
    for (int i = 0; i < 10; ++i) // only test 10 first elements
        for (int j = 0; j < elemSize; ++j)
            EXPECT_EQ(triangles1[i][j], triangles2[i][j]);


    //// create and get cross elements buffers
    const sofa::type::vector< TriangleSetTopologyContainer::EdgesInTriangle > & edgesInTriangle1 = topoCon->getEdgesInTriangleArray();
    const sofa::type::vector< TriangleSetTopologyContainer::TrianglesAroundVertex >& trianglesAroundVertex1 = topoCon->getTrianglesAroundVertexArray();
    const sofa::type::vector< TriangleSetTopologyContainer::TrianglesAroundEdge >& trianglesAroundEdge1 = topoCon->getTrianglesAroundEdgeArray();
    const TriangleSetTopologyContainer::SeqEdges& edges1 = topoCon->getEdges();

    const sofa::type::vector< BaseMeshTopology::EdgesInTriangle >& edgesInTriangle2 = topo->getEdgesInTriangleArray();
    const sofa::type::vector< BaseMeshTopology::TrianglesAroundVertex >& trianglesAroundVertex2 = topo->getTrianglesAroundVertexArray();
    const sofa::type::vector< BaseMeshTopology::TrianglesAroundEdge >& trianglesAroundEdge2= topo->getTrianglesAroundEdgeArray();
    const BaseMeshTopology::SeqEdges& edges2 = topo->getEdges();

    // check all buffers size
    EXPECT_EQ(edgesInTriangle1.size(), edgesInTriangle2.size());
    EXPECT_EQ(trianglesAroundVertex1.size(), trianglesAroundVertex2.size());
    EXPECT_EQ(trianglesAroundEdge1.size(), trianglesAroundEdge2.size());
    EXPECT_EQ(edges1.size(), edges2.size());
    EXPECT_EQ(topoCon->getNbPoints(), topo->getNbPoints());

    for (int i = 0; i < 6; ++i) // only test the 6 firsts elements of each buffer
    {
        const TriangleSetTopologyContainer::Edge& e1 = edges1[i];
        const BaseMeshTopology::Edge& e2 = edges2[i];
        for (int j = 0; j<2; ++j)
            EXPECT_EQ(e1[j], e2[j]);

        const TriangleSetTopologyContainer::EdgesInTriangle& EiT1 = edgesInTriangle1[i];
        const BaseMeshTopology::EdgesInTriangle& EiT2 = edgesInTriangle2[i];
        EXPECT_EQ(EiT1.size(), EiT2.size());
        for (size_t j=0; j<EiT1.size(); ++j)
            EXPECT_EQ(EiT1[j], EiT2[j]);

        const TriangleSetTopologyContainer::TrianglesAroundVertex& TaV1 = trianglesAroundVertex1[i];
        const BaseMeshTopology::TrianglesAroundVertex& TaV2 = trianglesAroundVertex2[i];
        EXPECT_EQ(TaV1.size(), TaV2.size());
        for (size_t j = 0; j<TaV1.size(); ++j)
            EXPECT_EQ(TaV1[j], TaV2[j]);

        const TriangleSetTopologyContainer::TrianglesAroundEdge& TaE1 = trianglesAroundEdge1[i];
        const BaseMeshTopology::TrianglesAroundEdge& TaE2 = trianglesAroundEdge2[i];
        EXPECT_EQ(TaE1.size(), TaE2.size());
        for (size_t j = 0; j<TaE1.size(); ++j)
            EXPECT_EQ(TaE1[j], TaE2[j]);
    }

    if (scene != nullptr)
        delete scene;
    if (scene2 != nullptr)
        delete scene2;

    return true;
}


bool MeshTopology_test::testEdgeTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_edges.obj", sofa::geometry::ElementType::EDGE);
    EdgeSetTopologyContainer* topoCon = dynamic_cast<EdgeSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    fake_TopologyScene* scene2 = new fake_TopologyScene("mesh/square1_edges.obj", sofa::geometry::ElementType::EDGE, true);
    MeshTopology* topo = dynamic_cast<MeshTopology*>(scene2->getNode().get()->getMeshTopology());
    topo->init();

    if (topoCon == nullptr || topo == nullptr)
    {
        if (scene != nullptr)
            delete scene;

        if (scene2 != nullptr)
            delete scene2;

        return false;
    }

    const int nbrEdge = 45;
    const int elemSize = 2;

    // Check edge container buffers size
    EXPECT_EQ(topo->getNbEdges(), nbrEdge);
    EXPECT_EQ(topo->getEdges().size(), nbrEdge);

    //// check edge buffer    
    const sofa::type::vector<EdgeSetTopologyContainer::Edge>& edges1 = topoCon->getEdgeArray();
    const sofa::core::topology::BaseMeshTopology::SeqEdges& edges2 = topo->getEdges();

    // Check edge buffer access    
    const EdgeSetTopologyContainer::Edge& edge1 = topo->getEdge(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(edge1[i], edges2[1][i]);

    // check buffer
    EXPECT_EQ(edges1.size(), edges2.size());
    for (int i = 0; i < 10; ++i) // only test 10 first elements
        for (int j = 0; j < elemSize; ++j)
            EXPECT_EQ(edges1[i][j], edges2[i][j]);


    // create and check vertex buffer
    const sofa::type::vector< EdgeSetTopologyContainer::EdgesAroundVertex >& edgeAroundVertices = topoCon->getEdgesAroundVertexArray();
    //TODO epernod 2018-07-05: access to full buffer does not exist in MeshTopology.
    for (int i = 0; i < 6; ++i) // only test first 6 arrays
    {
        const EdgeSetTopologyContainer::EdgesAroundVertex & edgeAV1 = edgeAroundVertices[i];
        const BaseMeshTopology::EdgesAroundVertex & edgeAV2 = topo->getEdgesAroundVertex(i);
        EXPECT_EQ(edgeAV1.size(), edgeAV2.size());

        for (size_t j = 0; j < edgeAV1.size(); ++j)
            EXPECT_EQ(edgeAV1[j], edgeAV2[j]);
    }

    if (scene != nullptr)
        delete scene;
    if (scene2 != nullptr)
        delete scene2;
    
    return true;
}


bool MeshTopology_test::testVertexTopology()
{
    
    return true;
}




TEST_F(MeshTopology_test, testEmptyContainer)
{
    ASSERT_TRUE(testEmptyContainer());
}

TEST_F(MeshTopology_test, testHexahedronTopology)
{
    ASSERT_TRUE(testHexahedronTopology());
}

TEST_F(MeshTopology_test, testTetrahedronTopology)
{
    ASSERT_TRUE(testTetrahedronTopology());
}

TEST_F(MeshTopology_test, testQuadTopology)
{
    ASSERT_TRUE(testQuadTopology());
}

TEST_F(MeshTopology_test, testTriangleTopology)
{
    ASSERT_TRUE(testTriangleTopology());
}

TEST_F(MeshTopology_test, testEdgeTopology)
{
    ASSERT_TRUE(testEdgeTopology());
}

// TODO epernod 2018-07-05
//TEST_F(MeshTopology_test, testVertexTopology)
//{
//    ASSERT_TRUE(testVertexTopology());
//}



// TODO epernod 2018-07-05: test element on Border
// TODO epernod 2018-07-05: test check connectivity
