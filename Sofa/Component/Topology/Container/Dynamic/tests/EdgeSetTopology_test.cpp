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

#include <sofa/testing/BaseTest.h>
#include <sofa/component/topology/testing/fake_TopologyScene.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyContainer.h>
#include <sofa/helper/system/FileRepository.h>


using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;


class EdgeSetTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();
    bool testEdgeBuffers();
    bool testVertexBuffers();
    bool checkTopology();
private:
    bool loadTopologyContainer(const std::string& filename);

    std::unique_ptr<fake_TopologyScene> m_scene;
    EdgeSetTopologyContainer::SPtr m_topoCon = nullptr;


    int nbrEdge = 45;
    int nbrVertex = 20;
};


bool EdgeSetTopology_test::loadTopologyContainer(const std::string& filename)
{
    m_scene = std::make_unique<fake_TopologyScene>(filename, sofa::core::topology::TopologyElementType::EDGE);

    if (m_scene == nullptr) {
        msg_error("EdgeSetTopology_test") << "Fake Topology creation failed.";
        return false;
    }

    auto root = m_scene->getNode().get();
    m_topoCon = root->get<EdgeSetTopologyContainer>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (m_topoCon == nullptr)
    {
        msg_error("EdgeSetTopology_test") << "EdgeSetTopologyContainer not found in scene.";
        return false;
    }

    return true;
}


bool EdgeSetTopology_test::testEmptyContainer()
{
    EdgeSetTopologyContainer::SPtr edgeContainer = sofa::core::objectmodel::New< EdgeSetTopologyContainer >();
    EXPECT_EQ(edgeContainer->getNbEdges(), 0);
    EXPECT_EQ(edgeContainer->getNumberOfElements(), 0);
    EXPECT_EQ(edgeContainer->getNumberOfEdges(), 0);
    EXPECT_EQ(edgeContainer->getEdges().size(), 0);
    
    EXPECT_EQ(edgeContainer->d_initPoints.getValue().size(), 0);
    EXPECT_EQ(edgeContainer->getNbPoints(), 0);

    return true;
}


bool EdgeSetTopology_test::testEdgeBuffers()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;
    
    // Check creation of the container
    EXPECT_EQ((m_topoCon->getName()), std::string("topoCon"));

    // Check edge container buffers size
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(m_topoCon->getNumberOfElements(), nbrEdge);
    EXPECT_EQ(m_topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(m_topoCon->getEdges().size(), nbrEdge);

    //// The first 2 edges in this file should be :
    sofa::type::fixed_array<EdgeSetTopologyContainer::PointID, 2> edgeTruth0(12, 17);
    sofa::type::fixed_array<EdgeSetTopologyContainer::PointID, 2> edgeTruth1(4, 12);


    //// check edge buffer
    const sofa::type::vector<EdgeSetTopologyContainer::Edge>& edges = m_topoCon->getEdgeArray();
    if (edges.empty())
        return false;
    
    // check edge 
    const EdgeSetTopologyContainer::Edge& edge0 = edges[0];
    EXPECT_EQ(edge0.size(), 2u);
    
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge0[i], edgeTruth0[i]);
    
    // TODO epernod 2018-07-05: method missing EdgeSetTopologyContainer::getVertexIndexInEdge
    // check edge indices
    //int vertexID = m_topoCon->getVertexIndexInEdge(tri0, triTruth0[1]);
    //EXPECT_EQ(vertexID, 1);
    //vertexID = m_topoCon->getVertexIndexInEdge(tri0, triTruth0[2]);
    //EXPECT_EQ(vertexID, 2);
    //vertexID = m_topoCon->getVertexIndexInEdge(tri0, 120);
    //EXPECT_EQ(vertexID, -1);


    // Check edge buffer access    
    const EdgeSetTopologyContainer::Edge& edge1 = m_topoCon->getEdge(1);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge1[i], edgeTruth1[i]);

    const EdgeSetTopologyContainer::Edge& edge2 = m_topoCon->getEdge(1000);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge2[i], sofa::InvalidID);

    return true;
}


bool EdgeSetTopology_test::testVertexBuffers()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;

    // create and check vertex buffer
    const sofa::type::vector< EdgeSetTopologyContainer::EdgesAroundVertex >& edgeAroundVertices = m_topoCon->getEdgesAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(m_topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex); 
    
    // check EdgesAroundVertex buffer access
    EXPECT_EQ(edgeAroundVertices.size(), nbrVertex);
    const EdgeSetTopologyContainer::EdgesAroundVertex& edgeAVertex = edgeAroundVertices[0];
    const EdgeSetTopologyContainer::EdgesAroundVertex& edgeAVertexM = m_topoCon->getEdgesAroundVertex(0);
    
    EXPECT_EQ(edgeAVertex.size(), edgeAVertexM.size());
    for (size_t i = 0; i < edgeAVertex.size(); i++)
        EXPECT_EQ(edgeAVertex[i], edgeAVertexM[i]);

    // check EdgesAroundVertex buffer element for this file    
    EXPECT_EQ(edgeAVertex[0], 26);
    EXPECT_EQ(edgeAVertex[1], 34);
    EXPECT_EQ(edgeAVertex[2], 41);

    return true;
}



bool EdgeSetTopology_test::checkTopology()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;

    bool res = m_topoCon->checkTopology();
    
    
    return res;
}



TEST_F(EdgeSetTopology_test, testEmptyContainer)
{
    ASSERT_TRUE(testEmptyContainer());
}

TEST_F(EdgeSetTopology_test, testEdgeBuffers)
{
    ASSERT_TRUE(testEdgeBuffers());
}

TEST_F(EdgeSetTopology_test, testVertexBuffers)
{
    ASSERT_TRUE(testVertexBuffers());
}

TEST_F(EdgeSetTopology_test, checkTopology)
{
    ASSERT_TRUE(checkTopology());
}


// TODO epernod 2018-07-05: test element on Border
// TODO epernod 2018-07-05: test Edge add/remove
// TODO epernod 2018-07-05: test check connectivity
