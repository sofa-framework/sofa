/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <sofa/helper/testing/BaseTest.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaSimulationGraph/SimpleApi.h>
#include <sofa/helper/system/FileRepository.h>
#include <SofaLoader/MeshObjLoader.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/fixed_array.h>

using namespace sofa::component::topology;
using namespace sofa::helper::testing;
using namespace sofa::simpleapi;
using namespace sofa::simpleapi::components;

class fake_EdgeTopologyScene
{
public:
    fake_EdgeTopologyScene(const std::string& filename)
        : m_edgeContainer(NULL)
        , m_filename(filename)
    {
        loadMesh();
    }

    bool loadMesh();

    EdgeSetTopologyContainer* m_edgeContainer;

protected:
    Simulation::SPtr m_simu;
    Node::SPtr m_root;

    std::string m_filename;
};


class EdgeSetTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();
    bool testEdgeBuffers();
    bool testVertexBuffers();
    bool checkTopology();
};


bool fake_EdgeTopologyScene::loadMesh()
{
    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");

    auto loader = createObject(m_root, "MeshObjLoader", {
        { "name","loader" },
        { "filename", sofa::helper::system::DataRepository.getFile(m_filename) } });

    auto topo = createObject(m_root, "EdgeSetTopologyContainer", {
        { "name", "topoCon" },
        { "edges", "@loader.edges" },
        { "position", "@loader.position" }
    });
   
    m_edgeContainer = dynamic_cast<EdgeSetTopologyContainer*>(topo.get());

    if (m_edgeContainer == NULL)
        return false;

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
    EXPECT_EQ(edgeContainer->getPoints().size(), 0);

    return true;
}


bool EdgeSetTopology_test::testEdgeBuffers()
{
    fake_EdgeTopologyScene* scene = new fake_EdgeTopologyScene("C:/projects/sofa-dev/share/mesh/square1_edges.obj");

    if (scene->m_edgeContainer == NULL)
    {
        if (scene != NULL)
            delete scene;
        return false;
    }

    // Check creation of the container
    EXPECT_EQ((scene->m_edgeContainer->getName()), std::string("topoCon"));

    // Check edge container buffers size
    EXPECT_EQ(scene->m_edgeContainer->getNbEdges(), 45);
    EXPECT_EQ(scene->m_edgeContainer->getNumberOfElements(), 45);
    EXPECT_EQ(scene->m_edgeContainer->getNumberOfEdges(), 45);
    EXPECT_EQ(scene->m_edgeContainer->getEdges().size(), 45);

    //// The first 2 edges in this file should be :
    sofa::helper::fixed_array<EdgeSetTopologyContainer::PointID, 2> edgeTruth0(12, 17);
    sofa::helper::fixed_array<EdgeSetTopologyContainer::PointID, 2> edgeTruth1(4, 12);


    //// check edge buffer
    const sofa::helper::vector<EdgeSetTopologyContainer::Edge>& edges = scene->m_edgeContainer->getEdgeArray();
    if (edges.empty())
        return false;
    
    // check edge 
    const EdgeSetTopologyContainer::Edge& edge0 = edges[0];
    EXPECT_EQ(edge0.size(), 2u);
    
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge0[i], edgeTruth0[i]);
    
    // TODO epernod 2018-07-05: method missing EdgeSetTopologyContainer::getVertexIndexInEdge
    // check edge indices
    //int vertexID = scene->m_edgeContainer->getVertexIndexInEdge(tri0, triTruth0[1]);
    //EXPECT_EQ(vertexID, 1);
    //vertexID = scene->m_edgeContainer->getVertexIndexInEdge(tri0, triTruth0[2]);
    //EXPECT_EQ(vertexID, 2);
    //vertexID = scene->m_edgeContainer->getVertexIndexInEdge(tri0, 120);
    //EXPECT_EQ(vertexID, -1);


    //// Check edge buffer access    
    const EdgeSetTopologyContainer::Edge& edge1 = scene->m_edgeContainer->getEdge(1);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge1[i], edgeTruth1[i]);

    const EdgeSetTopologyContainer::Edge& edge2 = scene->m_edgeContainer->getEdge(1000);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge2[i], -1);


    if(scene != NULL)
        delete scene;

    return true;
}


bool EdgeSetTopology_test::testVertexBuffers()
{
    fake_EdgeTopologyScene* scene = new fake_EdgeTopologyScene("C:/projects/sofa-dev/share/mesh/square1_edges.obj");

    if (scene->m_edgeContainer == NULL)
    {
        if (scene != NULL)
            delete scene;
        return false;
    }

    // create and check vertex buffer
    const sofa::helper::vector< EdgeSetTopologyContainer::EdgesAroundVertex >& edgeAroundVertices = scene->m_edgeContainer->getEdgesAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(scene->m_edgeContainer->d_initPoints.getValue().size(), 20);
    EXPECT_EQ(scene->m_edgeContainer->getNbPoints(), 20); 
    EXPECT_EQ(scene->m_edgeContainer->getPoints().size(), 20);


    // check EdgesAroundVertex buffer access
    EXPECT_EQ(edgeAroundVertices.size(), 20);
    const EdgeSetTopologyContainer::EdgesAroundVertex& edgeAVertex = edgeAroundVertices[0];
    const EdgeSetTopologyContainer::EdgesAroundVertex& edgeAVertexM = scene->m_edgeContainer->getEdgesAroundVertex(0);

    EXPECT_EQ(edgeAVertex.size(), edgeAVertexM.size());
    for (int i = 0; i < edgeAVertex.size(); i++)
        EXPECT_EQ(edgeAVertex[i], edgeAVertexM[i]);

    // check EdgesAroundVertex buffer element for this file    
    EXPECT_EQ(edgeAVertex[0], 26);
    EXPECT_EQ(edgeAVertex[1], 34);
    EXPECT_EQ(edgeAVertex[2], 41);

    return true;
}



bool EdgeSetTopology_test::checkTopology()
{
    fake_EdgeTopologyScene* scene = new fake_EdgeTopologyScene("C:/projects/sofa-dev/share/mesh/square1_edges.obj");

    if (scene->m_edgeContainer == NULL)
    {
        if (scene != NULL)
            delete scene;
        return false;
    }

    bool res = scene->m_edgeContainer->checkTopology();
    
    if (scene != NULL)
        delete scene;
    
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
