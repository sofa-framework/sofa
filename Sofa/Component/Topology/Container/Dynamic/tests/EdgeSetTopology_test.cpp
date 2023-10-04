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
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>
#include <sofa/core/topology/TopologyHandler.h>
#include <sofa/helper/system/FileRepository.h>


using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;

/// <summary>
/// Class to test @sa EdgeSetTopologyContainer and @sa EdgeSetTopologyModifier methods
/// </summary>
class EdgeSetTopology_test : public BaseTest
{
public:
    using Edge = EdgeSetTopologyContainer::Edge;
    using EdgeID = EdgeSetTopologyContainer::EdgeID;
    using EdgesAroundVertex = EdgeSetTopologyContainer::EdgesAroundVertex;

    /// Method to test @sa EdgeSetTopologyContainer creation and initialisation buffers without inputs.
    bool testEmptyContainer();

    /// Method to test @sa EdgeSetTopologyContainer Edge creation and initialisation with a mesh as input.
    bool testEdgeBuffers();

    /// Method to test @sa EdgeSetTopologyContainer EdgesAroundVertex creation and initialisation with a mesh as input.
    bool testVertexBuffers();

    /// Method to test @sa EdgeSetTopologyContainer checkTopology method.
    bool checkTopology();


    /// Method to test @sa EdgeSetTopologyModifier removeVertices method and check edge buffers.
    bool testRemovingVertices();

    /// Method to test @sa EdgeSetTopologyModifier removeEdges method with isolated vertices and check edge buffers.
    bool testRemovingEdges();

    /// Method to test @sa EdgeSetTopologyModifier addEdges method and check edge buffers.
    bool testAddingEdges();

    /// Method to check EdgeSetTopologyContainer list of TopologyHandlers and TopologyData from other components.
    bool checkEdgeDataGraph();
   

private:
    /// <summary>
    /// Method to factorize the creation and loading of the @sa m_scene and retrieve Topology container @sa m_topoCon
    /// </summary>
    /// <param name="filename">Path string of the mesh to load</param>
    /// <returns>Bool True if loaded success</returns>
    bool loadTopologyContainer(const std::string& filename);

    /// Pointer to the basic scene created with a topology for the tests
    std::unique_ptr<fake_TopologyScene> m_scene;
    
    /// Pointer to the topology container created in the scene @sa m_scene
    EdgeSetTopologyContainer::SPtr m_topoCon = nullptr;

    /// GroundTruth values of reference mesh used in test: square1_edges.obj
    int nbrEdge = 45;
    int nbrVertex = 20;
};


bool EdgeSetTopology_test::loadTopologyContainer(const std::string& filename)
{
    m_scene = std::make_unique<fake_TopologyScene>(filename, sofa::geometry::ElementType::EDGE);

    if (m_scene == nullptr) {
        msg_error("EdgeSetTopology_test") << "Fake Topology creation failed.";
        return false;
    }

    const auto root = m_scene->getNode().get();
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
    const EdgeSetTopologyContainer::SPtr edgeContainer = sofa::core::objectmodel::New< EdgeSetTopologyContainer >();
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
    const sofa::type::vector<Edge>& edges = m_topoCon->getEdgeArray();
    if (edges.empty())
        return false;
    
    // check edge 
    const Edge& edge0 = edges[0];
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
    const Edge& edge1 = m_topoCon->getEdge(1);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge1[i], edgeTruth1[i]);

    const Edge& edge2 = m_topoCon->getEdge(1000);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge2[i], sofa::InvalidID);

    return true;
}


bool EdgeSetTopology_test::testVertexBuffers()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;

    // create and check vertex buffer
    const sofa::type::vector< EdgesAroundVertex >& edgeAroundVertices = m_topoCon->getEdgesAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(m_topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex); 
    
    // check EdgesAroundVertex buffer access
    EXPECT_EQ(edgeAroundVertices.size(), nbrVertex);
    const EdgesAroundVertex& edgeAVertex = edgeAroundVertices[0];
    const EdgesAroundVertex& edgeAVertexM = m_topoCon->getEdgesAroundVertex(0);
    
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

    const bool res = m_topoCon->checkTopology();
    
    return res;
}


bool EdgeSetTopology_test::testRemovingVertices()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;

    // Check edge buffer access
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex);
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge);

    // Get access to the Edge modifier
    const auto root = m_scene->getNode().get();
    const EdgeSetTopologyModifier::SPtr edgeModifier = root->get<EdgeSetTopologyModifier>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (edgeModifier == nullptr)
        return false;

    // Check edge around point to be removed
    sofa::type::vector< EdgeID > vIds = { 0, 1, 2 };
    //const EdgesAroundVertex& edgeAVertex = m_topoCon->getEdgesAroundVertex(0);

    edgeModifier->removePoints(vIds);

    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex - vIds.size());
    // TODO epernod 2022-08-24: Edge are not deleted when removing vertices. This might create errors.
    //auto nbrE = edgeAVertex.size();
    //EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge - nbrE); 

    return true;
}


bool EdgeSetTopology_test::testRemovingEdges()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;

    // Check edge buffer access
    const sofa::type::vector<Edge>& edges = m_topoCon->getEdgeArray();
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex);
    EXPECT_EQ(edges.size(), nbrEdge);

    // Get access to the Edge modifier
    const auto root = m_scene->getNode().get();
    const EdgeSetTopologyModifier::SPtr edgeModifier = root->get<EdgeSetTopologyModifier>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (edgeModifier == nullptr)
        return false;

    // Check first the swap + pop_back method
    Edge lastEdge = edges.back();
    const sofa::type::vector< EdgeID > edgeIds = { 0 };
    
    // Remove first edge from the buffer
    edgeModifier->removeEdges(edgeIds);
    
    // Check size of the new edge buffer
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge - 1);

    // Check that first edge is now the previous last edge
    Edge newEdge = m_topoCon->getEdge(0);
    EXPECT_EQ(lastEdge[0], newEdge[0]);
    EXPECT_EQ(lastEdge[1], newEdge[1]);


    // Check isolate vertex removal
    const EdgesAroundVertex& edgeALastVertex = m_topoCon->getEdgesAroundVertex(nbrVertex - 1);
    const auto nbr = nbrEdge - 1 - edgeALastVertex.size();

    edgeModifier->removeEdges(edgeALastVertex);
    
    EXPECT_EQ(m_topoCon->getNbEdges(), nbr);
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex - 1);

    return true;
}


bool EdgeSetTopology_test::testAddingEdges()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;

    // Check edge buffer access
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex);
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge);

    // Get access to the Edge modifier
    const auto root = m_scene->getNode().get();
    const EdgeSetTopologyModifier::SPtr edgeModifier = root->get<EdgeSetTopologyModifier>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (edgeModifier == nullptr)
        return false;

    sofa::type::vector< Edge > edgesToAdd;
    edgesToAdd.push_back(Edge(0, 5));
    
    // Add edges
    edgeModifier->addEdges(edgesToAdd);
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge + edgesToAdd.size());

    // Add same edge again
    edgeModifier->addEdges(edgesToAdd);
    // TODO epernod 2022-08-24: There is no check to add duplicated edges
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge + edgesToAdd.size() * 2); 

    return true;
}


bool EdgeSetTopology_test::checkEdgeDataGraph()
{
    if (!loadTopologyContainer("mesh/square1_edges.obj"))
        return false;
    
    // Get the number of TopologyData linked to the topology buffer (one from Mass and one from VectorSpringFF)
    auto& outputs = m_topoCon->d_edge.getOutputs();
    EXPECT_EQ(outputs.size(), 2);

    auto edgeHandlers = m_topoCon->getTopologyHandlerList(sofa::geometry::ElementType::EDGE);
    const auto vertexHandlers = m_topoCon->getTopologyHandlerList(sofa::geometry::ElementType::POINT);
    
    EXPECT_EQ(vertexHandlers.size(), 1);
    EXPECT_EQ(edgeHandlers.size(), 2);

    sofa::core::topology::TopologyHandler* vertexH0 = *vertexHandlers.cbegin();
    EXPECT_NE(vertexH0, nullptr);
    EXPECT_EQ(vertexH0->getName(), "TopologyDataHandler (MeshMatrixMass)vertexMass");

    auto itHandler = edgeHandlers.begin();
    sofa::core::topology::TopologyHandler* edgeH0 = *itHandler;
    EXPECT_NE(edgeH0, nullptr);

    // We need to pre-check as handlers are stored in a set (order not predefined)
    const bool firstOrder = edgeH0->getName().find("edgeMass") != std::string::npos;

    if (firstOrder)
        EXPECT_EQ(edgeH0->getName(), "TopologyDataHandler (MeshMatrixMass)edgeMass");
    else
        EXPECT_EQ(edgeH0->getName(), "TopologyDataHandler (VectorSpringForceField)springs");

    itHandler++;
    sofa::core::topology::TopologyHandler* edgeH1 = *itHandler;
    EXPECT_NE(edgeH1, nullptr);

    if (firstOrder)
        EXPECT_EQ(edgeH1->getName(), "TopologyDataHandler (VectorSpringForceField)springs");
    else
        EXPECT_EQ(edgeH1->getName(), "TopologyDataHandler (MeshMatrixMass)edgeMass");
        
    return true;
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


TEST_F(EdgeSetTopology_test, testRemovingVertices)
{
    ASSERT_TRUE(testRemovingVertices());
}

TEST_F(EdgeSetTopology_test, testRemovingEdges)
{
    ASSERT_TRUE(testRemovingEdges());
}

TEST_F(EdgeSetTopology_test, testAddingEdges)
{
    ASSERT_TRUE(testAddingEdges());
}

TEST_F(EdgeSetTopology_test, checkEdgeDataGraph)
{
    ASSERT_TRUE(checkEdgeDataGraph());
}

// TODO epernod 2018-07-05: test element on Border
// TODO epernod 2018-07-05: test check connectivity
