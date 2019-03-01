/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseTopology/SofaBaseTopology_test/fake_TopologyScene.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <sofa/helper/system/FileRepository.h>


using namespace sofa::component::topology;
using namespace sofa::helper::testing;


class EdgeSetTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();
    bool testEdgeBuffers();
    bool testVertexBuffers();
    bool checkTopology();

    int nbrEdge = 45;
    int nbrVertex = 20;
};


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
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_edges.obj", sofa::core::topology::TopologyObjectType::EDGE);
    EdgeSetTopologyContainer* topoCon = dynamic_cast<EdgeSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == NULL)
    {
        if (scene != NULL)
            delete scene;
        return false;
    }

    // Check creation of the container
    EXPECT_EQ((topoCon->getName()), std::string("topoCon"));

    // Check edge container buffers size
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNumberOfElements(), nbrEdge);
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    //// The first 2 edges in this file should be :
    sofa::helper::fixed_array<EdgeSetTopologyContainer::PointID, 2> edgeTruth0(12, 17);
    sofa::helper::fixed_array<EdgeSetTopologyContainer::PointID, 2> edgeTruth1(4, 12);


    //// check edge buffer
    const sofa::helper::vector<EdgeSetTopologyContainer::Edge>& edges = topoCon->getEdgeArray();
    if (edges.empty())
        return false;
    
    // check edge 
    const EdgeSetTopologyContainer::Edge& edge0 = edges[0];
    EXPECT_EQ(edge0.size(), 2u);
    
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge0[i], edgeTruth0[i]);
    
    // TODO epernod 2018-07-05: method missing EdgeSetTopologyContainer::getVertexIndexInEdge
    // check edge indices
    //int vertexID = topoCon->getVertexIndexInEdge(tri0, triTruth0[1]);
    //EXPECT_EQ(vertexID, 1);
    //vertexID = topoCon->getVertexIndexInEdge(tri0, triTruth0[2]);
    //EXPECT_EQ(vertexID, 2);
    //vertexID = topoCon->getVertexIndexInEdge(tri0, 120);
    //EXPECT_EQ(vertexID, -1);


    // Check edge buffer access    
    const EdgeSetTopologyContainer::Edge& edge1 = topoCon->getEdge(1);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge1[i], edgeTruth1[i]);

    const EdgeSetTopologyContainer::Edge& edge2 = topoCon->getEdge(1000);
    for (int i = 0; i<2; ++i)
        EXPECT_EQ(edge2[i], -1);


    if(scene != NULL)
        delete scene;

    return true;
}


bool EdgeSetTopology_test::testVertexBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_edges.obj", sofa::core::topology::TopologyObjectType::EDGE);
    EdgeSetTopologyContainer* topoCon = dynamic_cast<EdgeSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == NULL)
    {
        if (scene != NULL)
            delete scene;
        return false;
    }

    // create and check vertex buffer
    const sofa::helper::vector< EdgeSetTopologyContainer::EdgesAroundVertex >& edgeAroundVertices = topoCon->getEdgesAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(topoCon->getNbPoints(), nbrVertex); 
    EXPECT_EQ(topoCon->getPoints().size(), nbrVertex);


    // check EdgesAroundVertex buffer access
    EXPECT_EQ(edgeAroundVertices.size(), nbrVertex);
    const EdgeSetTopologyContainer::EdgesAroundVertex& edgeAVertex = edgeAroundVertices[0];
    const EdgeSetTopologyContainer::EdgesAroundVertex& edgeAVertexM = topoCon->getEdgesAroundVertex(0);
    
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
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_edges.obj", sofa::core::topology::TopologyObjectType::EDGE);
    EdgeSetTopologyContainer* topoCon = dynamic_cast<EdgeSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == NULL)
    {
        if (scene != NULL)
            delete scene;
        return false;
    }

    bool res = topoCon->checkTopology();
    
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
