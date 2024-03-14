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
#include <sofa/component/topology/container/dynamic/QuadSetTopologyContainer.h>
#include <sofa/helper/system/FileRepository.h>

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;


class QuadSetTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();
    bool testQuadBuffers();
    bool testEdgeBuffers();
    bool testVertexBuffers();
    bool checkTopology();

    // ground truth from obj file;
    int nbrQuad = 9;
    int nbrEdge = 24;
    int nbrVertex = 16;
    int elemSize = 4;
};


bool QuadSetTopology_test::testEmptyContainer()
{
    const QuadSetTopologyContainer::SPtr topoCon = sofa::core::objectmodel::New< QuadSetTopologyContainer >();
    EXPECT_EQ(topoCon->getNbQuads(), 0);
    EXPECT_EQ(topoCon->getNumberOfElements(), 0);
    EXPECT_EQ(topoCon->getNumberOfQuads(), 0);
    EXPECT_EQ(topoCon->getQuads().size(), 0);

    EXPECT_EQ(topoCon->getNumberOfEdges(), 0);
    EXPECT_EQ(topoCon->getNbEdges(), 0);
    EXPECT_EQ(topoCon->getEdges().size(), 0);

    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), 0);
    EXPECT_EQ(topoCon->getNbPoints(), 0);
    
    return true;
}


bool QuadSetTopology_test::testQuadBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_quads.obj", sofa::geometry::ElementType::QUAD);
    QuadSetTopologyContainer* topoCon = dynamic_cast<QuadSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // Check creation of the container
    EXPECT_EQ((topoCon->getName()), std::string("topoCon"));

    // Check topology element container buffers size
    EXPECT_EQ(topoCon->getNbQuads(), nbrQuad);
    EXPECT_EQ(topoCon->getNumberOfElements(), nbrQuad);
    EXPECT_EQ(topoCon->getNumberOfQuads(), nbrQuad);
    EXPECT_EQ(topoCon->getQuads().size(), nbrQuad);

    // check edges buffer has been created
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    // The first 2 elements in this file should be :
    sofa::type::fixed_array<QuadSetTopologyContainer::PointID, 4> elemTruth0(3, 11, 5, 4);
    sofa::type::fixed_array<QuadSetTopologyContainer::PointID, 4> elemTruth1(11, 8, 6, 5);


    // check topology element buffer
    const sofa::type::vector<QuadSetTopologyContainer::Quad>& elements = topoCon->getQuadArray();
    if (elements.empty())
        return false;
    
    // check topology element 
    const QuadSetTopologyContainer::Quad& elem0 = elements[0];
    EXPECT_EQ(elem0.size(), 4u);
    
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem0[i], elemTruth0[i]);
    
    // check topology element indices
    int vertexID = topoCon->getVertexIndexInQuad(elem0, elemTruth0[1]);
    EXPECT_EQ(vertexID, 1);
    vertexID = topoCon->getVertexIndexInQuad(elem0, elemTruth0[2]);
    EXPECT_EQ(vertexID, 2);
    vertexID = topoCon->getVertexIndexInQuad(elem0, 120);
    EXPECT_EQ(vertexID, -1);


    // Check topology element buffer access    
    const QuadSetTopologyContainer::Quad& elem1 = topoCon->getQuad(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem1[i], elemTruth1[i]);

    const QuadSetTopologyContainer::Quad& elem2 = topoCon->getQuad(1000);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem2[i], sofa::InvalidID);


    if(scene != nullptr)
        delete scene;

    return true;
}


bool QuadSetTopology_test::testEdgeBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_quads.obj", sofa::geometry::ElementType::QUAD);
    QuadSetTopologyContainer* topoCon = dynamic_cast<QuadSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check edges
    const sofa::type::vector< QuadSetTopologyContainer::QuadsAroundEdge >& elemAroundEdges = topoCon->getQuadsAroundEdgeArray();
        
    // check only the edge buffer size: Full test on edges are done in EdgeSetTopology_test
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    // check edge created element
    QuadSetTopologyContainer::Edge edge = topoCon->getEdge(0);
    EXPECT_EQ(edge[0], 5);
    EXPECT_EQ(edge[1], 11);


    // check QuadAroundEdge buffer access
    EXPECT_EQ(elemAroundEdges.size(), nbrEdge);
    const QuadSetTopologyContainer::QuadsAroundEdge& elemAEdge = elemAroundEdges[0];
    const QuadSetTopologyContainer::QuadsAroundEdge& elemAEdgeM = topoCon->getQuadsAroundEdge(0);

    EXPECT_EQ(elemAEdge.size(), elemAEdgeM.size());
    for (size_t i = 0; i < elemAEdge.size(); i++)
        EXPECT_EQ(elemAEdge[i], elemAEdgeM[i]);

    // check QuadAroundEdge buffer element for this file
    EXPECT_EQ(elemAEdge[0], 0);
    EXPECT_EQ(elemAEdge[1], 1);


    // check EdgesInQuad buffer acces
    const sofa::type::vector< QuadSetTopologyContainer::EdgesInQuad > & edgeInQuads = topoCon->getEdgesInQuadArray();
    EXPECT_EQ(edgeInQuads.size(), nbrQuad);

    const QuadSetTopologyContainer::EdgesInQuad& edgeInElem = edgeInQuads[2];
    const QuadSetTopologyContainer::EdgesInQuad& edgeInElemM = topoCon->getEdgesInQuad(2);

    EXPECT_EQ(edgeInElem.size(), edgeInElemM.size());
    for (size_t i = 0; i < edgeInElem.size(); i++)
        EXPECT_EQ(edgeInElem[i], edgeInElemM[i]);

    sofa::type::fixed_array<int, 4> edgeInElemTruth(7, 8, 4, 9);
    for (int i = 0; i<4; ++i)
        EXPECT_EQ(edgeInElem[i], edgeInElemTruth[i]);
    
    
    // Check Edge Index in Quad
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(topoCon->getEdgeIndexInQuad(edgeInElem, edgeInElemTruth[i]), i);

    int edgeId = topoCon->getEdgeIndexInQuad(edgeInElem, 20000);
    EXPECT_EQ(edgeId, -1);

    // check link between QuadsAroundEdge and EdgesInQuad
    for (unsigned int i = 0; i < 4; ++i)
    {
        QuadSetTopologyContainer::EdgeID edgeId = i;
        const QuadSetTopologyContainer::QuadsAroundEdge& _elemAEdge = elemAroundEdges[edgeId];
        for (size_t j = 0; j < _elemAEdge.size(); ++j)
        {
            QuadSetTopologyContainer::QuadID triId = _elemAEdge[j];
            const QuadSetTopologyContainer::EdgesInQuad& _edgeInElem = edgeInQuads[triId];
            bool found = false;
            for (size_t k = 0; k < _edgeInElem.size(); ++k)
            {
                if (_edgeInElem[k] == edgeId)
                {
                    found = true;
                    break;
                }
            }

            if (found == false)
            {
                if (scene != nullptr)
                    delete scene;
                return false;
            }
        }
    }
    
    return true;
}


bool QuadSetTopology_test::testVertexBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_quads.obj", sofa::geometry::ElementType::QUAD);
    QuadSetTopologyContainer* topoCon = dynamic_cast<QuadSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check vertex buffer
    const sofa::type::vector< QuadSetTopologyContainer::QuadsAroundVertex >& elemAroundVertices = topoCon->getQuadsAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(topoCon->getNbPoints(), nbrVertex); //TODO: check why 0 and not 20
    
    // check QuadsAroundVertex buffer access
    EXPECT_EQ(elemAroundVertices.size(), nbrVertex);
    const QuadSetTopologyContainer::QuadsAroundVertex& elemAVertex = elemAroundVertices[1];
    const QuadSetTopologyContainer::QuadsAroundVertex& elemAVertexM = topoCon->getQuadsAroundVertex(1);

    EXPECT_EQ(elemAVertex.size(), elemAVertexM.size());
    for (size_t i = 0; i < elemAVertex.size(); i++)
        EXPECT_EQ(elemAVertex[i], elemAVertexM[i]);

    // check QuadsAroundVertex buffer element for this file    
    EXPECT_EQ(elemAVertex[0], 6);
    EXPECT_EQ(elemAVertex[1], 7);
    

    // Check VertexIndexInQuad
    const QuadSetTopologyContainer::Quad &elem = topoCon->getQuad(1);
    int vId = topoCon->getVertexIndexInQuad(elem, elem[0]);
    EXPECT_NE(vId, -1);
    vId = topoCon->getVertexIndexInQuad(elem, 200000);
    EXPECT_EQ(vId, -1);

    return true;
}



bool QuadSetTopology_test::checkTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/square1_quads.obj", sofa::geometry::ElementType::QUAD);
    const QuadSetTopologyContainer* topoCon = dynamic_cast<QuadSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    const bool res = topoCon->checkTopology();
    
    if (scene != nullptr)
        delete scene;
    
    return res;
}



TEST_F(QuadSetTopology_test, testEmptyContainer)
{
    ASSERT_TRUE(testEmptyContainer());
}

TEST_F(QuadSetTopology_test, testQuadBuffers)
{
    ASSERT_TRUE(testQuadBuffers());
}

TEST_F(QuadSetTopology_test, testEdgeBuffers)
{
    ASSERT_TRUE(testEdgeBuffers());
}

TEST_F(QuadSetTopology_test, testVertexBuffers)
{
    ASSERT_TRUE(testVertexBuffers());
}

TEST_F(QuadSetTopology_test, checkTopology)
{
    ASSERT_TRUE(checkTopology());
}


// TODO epernod 2018-07-05: test element on Border
// TODO epernod 2018-07-05: test quad add/remove
// TODO epernod 2018-07-05: test check connectivity
