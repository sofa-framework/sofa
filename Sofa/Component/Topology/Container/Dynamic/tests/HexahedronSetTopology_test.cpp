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
#include <sofa/component/topology/container/dynamic/HexahedronSetTopologyContainer.h>
#include <sofa/helper/system/FileRepository.h>

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;


class HexahedronSetTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();
    bool testHexahedronBuffers();
    bool testQuadBuffers();
    bool testEdgeBuffers();
    bool testVertexBuffers();
    bool checkTopology();

    // ground truth from obj file;
    int nbrHexahedron = 9;
    int nbrQuad = 42;
    int nbrEdge = 64;
    int nbrVertex = 32;
    int elemSize = 8;
};


bool HexahedronSetTopology_test::testEmptyContainer()
{
    const HexahedronSetTopologyContainer::SPtr topoCon = sofa::core::objectmodel::New< HexahedronSetTopologyContainer >();
    EXPECT_EQ(topoCon->getNbHexahedra(), 0);
    EXPECT_EQ(topoCon->getNumberOfElements(), 0);
    EXPECT_EQ(topoCon->getNumberOfHexahedra(), 0);
    EXPECT_EQ(topoCon->getHexahedra().size(), 0);

    EXPECT_EQ(topoCon->getNumberOfQuads(), 0);
    EXPECT_EQ(topoCon->getNbQuads(), 0);
    EXPECT_EQ(topoCon->getQuads().size(), 0);

    EXPECT_EQ(topoCon->getNumberOfEdges(), 0);
    EXPECT_EQ(topoCon->getNbEdges(), 0);
    EXPECT_EQ(topoCon->getEdges().size(), 0);

    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), 0);
    EXPECT_EQ(topoCon->getNbPoints(), 0);

    return true;
}


bool HexahedronSetTopology_test::testHexahedronBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON);
    HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // Check creation of the container
    EXPECT_EQ((topoCon->getName()), std::string("topoCon"));

    // Check topology element container buffers size
    EXPECT_EQ(topoCon->getNbHexahedra(), nbrHexahedron);
    EXPECT_EQ(topoCon->getNumberOfElements(), nbrHexahedron);
    EXPECT_EQ(topoCon->getNumberOfHexahedra(), nbrHexahedron);
    EXPECT_EQ(topoCon->getHexahedra().size(), nbrHexahedron);

    // check quads buffer has been created
    EXPECT_EQ(topoCon->getNumberOfQuads(), nbrQuad);
    EXPECT_EQ(topoCon->getNbQuads(), nbrQuad);
    EXPECT_EQ(topoCon->getQuads().size(), nbrQuad);

    // check edges buffer has been created
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    // The first 2 elements in this file should be :
    sofa::type::fixed_array<HexahedronSetTopologyContainer::PointID, 8> elemTruth0(0, 1, 5, 4, 16, 17, 21, 20);
    sofa::type::fixed_array<HexahedronSetTopologyContainer::PointID, 8> elemTruth1(1, 2, 6, 5, 17, 18, 22, 21);


    // check topology element buffer
    const sofa::type::vector<HexahedronSetTopologyContainer::Hexahedron>& elements = topoCon->getHexahedronArray();
    if (elements.empty())
        return false;
    
    // check topology element 
    const HexahedronSetTopologyContainer::Hexahedron& elem0 = elements[0];
    EXPECT_EQ(elem0.size(), 8u);
    
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem0[i], elemTruth0[i]);
    
    // check topology element indices
    int vertexID = topoCon->getVertexIndexInHexahedron(elem0, elemTruth0[1]);
    EXPECT_EQ(vertexID, 1);
    vertexID = topoCon->getVertexIndexInHexahedron(elem0, elemTruth0[2]);
    EXPECT_EQ(vertexID, 2);
    vertexID = topoCon->getVertexIndexInHexahedron(elem0, 12000);
    EXPECT_EQ(vertexID, -1);


    // Check topology element buffer access    
    const HexahedronSetTopologyContainer::Hexahedron& elem1 = topoCon->getHexahedron(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem1[i], elemTruth1[i]);

    const HexahedronSetTopologyContainer::Hexahedron& elem2 = topoCon->getHexahedron(100000);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem2[i], sofa::InvalidID);


    if(scene != nullptr)
        delete scene;

    return true;
}


bool HexahedronSetTopology_test::testQuadBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON);
    HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check quads
    const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundQuad >& elemAroundQuads = topoCon->getHexahedraAroundQuadArray();

    // check only the quad buffer size: Full test on quads are done in QuadSetTopology_test
    EXPECT_EQ(topoCon->getNumberOfQuads(), nbrQuad);
    EXPECT_EQ(topoCon->getNbQuads(), nbrQuad);
    EXPECT_EQ(topoCon->getQuads().size(), nbrQuad);


    // check quad created element
    HexahedronSetTopologyContainer::Quad quad = topoCon->getQuad(0);
    EXPECT_EQ(quad[0], 0);
    EXPECT_EQ(quad[1], 4);
    EXPECT_EQ(quad[2], 5);
    EXPECT_EQ(quad[3], 1);


    // check HexahedraAroundQuad buffer access
    EXPECT_EQ(elemAroundQuads.size(), nbrQuad);
    const HexahedronSetTopologyContainer::HexahedraAroundQuad& elemAQuad = elemAroundQuads[3];
    const HexahedronSetTopologyContainer::HexahedraAroundQuad& elemAQuadM = topoCon->getHexahedraAroundQuad(3);

    EXPECT_EQ(elemAQuad.size(), elemAQuadM.size());
    for (size_t i = 0; i < elemAQuad.size(); i++)
        EXPECT_EQ(elemAQuad[i], elemAQuadM[i]);

    // check HexahedraAroundQuad buffer element for this file
    EXPECT_EQ(elemAQuad[0], 0);
    EXPECT_EQ(elemAQuad[1], 1);


    // check QuadsInHexahedron buffer acces
    const sofa::type::vector< HexahedronSetTopologyContainer::QuadsInHexahedron > & quadInHexahedra = topoCon->getQuadsInHexahedronArray();
    EXPECT_EQ(quadInHexahedra.size(), nbrHexahedron);

    const HexahedronSetTopologyContainer::QuadsInHexahedron& quadInElem = quadInHexahedra[1];
    const HexahedronSetTopologyContainer::QuadsInHexahedron& quadInElemM = topoCon->getQuadsInHexahedron(1);

    EXPECT_EQ(quadInElem.size(), quadInElemM.size());
    for (size_t i = 0; i < quadInElem.size(); i++)
        EXPECT_EQ(quadInElem[i], quadInElemM[i]);

    sofa::type::fixed_array<int, 6> quadInElemTruth(6, 7, 8, 9, 10, 3);
    for (size_t i = 0; i<quadInElemTruth.size(); ++i)
        EXPECT_EQ(quadInElem[i], quadInElemTruth[i]);


    // Check Quad Index in Hexahedron
    for (size_t i = 0; i<quadInElemTruth.size(); ++i)
        EXPECT_EQ(topoCon->getQuadIndexInHexahedron(quadInElem, quadInElemTruth[i]), i);

    int quadId = topoCon->getQuadIndexInHexahedron(quadInElem, 20000);
    EXPECT_EQ(quadId, -1);

    // check link between HexahedraAroundQuad and QuadsInHexahedron
    for (unsigned int i = 0; i < 4; ++i)
    {
        HexahedronSetTopologyContainer::QuadID quadId = i;
        const HexahedronSetTopologyContainer::HexahedraAroundQuad& _elemAQuad = elemAroundQuads[quadId];
        for (size_t j = 0; j < _elemAQuad.size(); ++j)
        {
            HexahedronSetTopologyContainer::HexahedronID triId = _elemAQuad[j];
            const HexahedronSetTopologyContainer::QuadsInHexahedron& _quadInElem = quadInHexahedra[triId];
            bool found = false;
            for (size_t k = 0; k < _quadInElem.size(); ++k)
            {
                if (_quadInElem[k] == quadId)
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


bool HexahedronSetTopology_test::testEdgeBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON);
    HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check edges
    const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundEdge >& elemAroundEdges = topoCon->getHexahedraAroundEdgeArray();
        
    // check only the edge buffer size: Full test on edges are done in EdgeSetTopology_test
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    // check edge created element
    HexahedronSetTopologyContainer::Edge edge = topoCon->getEdge(0);
    EXPECT_EQ(edge[0], 4);
    EXPECT_EQ(edge[1], 5);


    // check HexahedronAroundEdge buffer access
    EXPECT_EQ(elemAroundEdges.size(), nbrEdge);
    const HexahedronSetTopologyContainer::HexahedraAroundEdge& elemAEdge = elemAroundEdges[13];
    const HexahedronSetTopologyContainer::HexahedraAroundEdge& elemAEdgeM = topoCon->getHexahedraAroundEdge(13);

    EXPECT_EQ(elemAEdge.size(), elemAEdgeM.size());
    for (size_t i = 0; i < elemAEdge.size(); i++)
        EXPECT_EQ(elemAEdge[i], elemAEdgeM[i]);

    // check HexahedronAroundEdge buffer element for this file
    EXPECT_EQ(elemAEdge[0], 1);
    EXPECT_EQ(elemAEdge[1], 2);


    // check EdgesInHexahedron buffer acces
    const sofa::type::vector< HexahedronSetTopologyContainer::EdgesInHexahedron > & edgeInHexahedra = topoCon->getEdgesInHexahedronArray();
    EXPECT_EQ(edgeInHexahedra.size(), nbrHexahedron);

    const HexahedronSetTopologyContainer::EdgesInHexahedron& edgeInElem = edgeInHexahedra[2];
    const HexahedronSetTopologyContainer::EdgesInHexahedron& edgeInElemM = topoCon->getEdgesInHexahedron(2);

    EXPECT_EQ(edgeInElem.size(), edgeInElemM.size());
    for (size_t i = 0; i < edgeInElem.size(); i++)
        EXPECT_EQ(edgeInElem[i], edgeInElemM[i]);
    
    sofa::type::fixed_array<int, 10> edgeInElemTruth(22, 13, 18, 21, 26, 20, 27, 19, 25, 15); // Test only 10 out of 12 edges as no fixed_array<12>
    for (size_t i = 0; i<edgeInElemTruth.size(); ++i)
        EXPECT_EQ(edgeInElem[i], edgeInElemTruth[i]);
    
    
    // Check Edge Index in Hexahedron
    for (size_t i = 0; i<edgeInElemTruth.size(); ++i)
        EXPECT_EQ(topoCon->getEdgeIndexInHexahedron(edgeInElem, edgeInElemTruth[i]), i);

    int edgeId = topoCon->getEdgeIndexInHexahedron(edgeInElem, 20000);
    EXPECT_EQ(edgeId, -1);

    // check link between HexahedraAroundEdge and EdgesInHexahedron
    for (int i = 0; i < 4; ++i)
    {
        HexahedronSetTopologyContainer::EdgeID edgeId = i;
        const HexahedronSetTopologyContainer::HexahedraAroundEdge& _elemAEdge = elemAroundEdges[edgeId];
        for (size_t j = 0; j < _elemAEdge.size(); ++j)
        {
            HexahedronSetTopologyContainer::HexahedronID triId = _elemAEdge[j];
            const HexahedronSetTopologyContainer::EdgesInHexahedron& _edgeInElem = edgeInHexahedra[triId];
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


bool HexahedronSetTopology_test::testVertexBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON);
    HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check vertex buffer
    const sofa::type::vector< HexahedronSetTopologyContainer::HexahedraAroundVertex >& elemAroundVertices = topoCon->getHexahedraAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(topoCon->getNbPoints(), nbrVertex); //TODO: check why 0 and not 20
   
    // check HexahedraAroundVertex buffer access
    EXPECT_EQ(elemAroundVertices.size(), nbrVertex);
    const HexahedronSetTopologyContainer::HexahedraAroundVertex& elemAVertex = elemAroundVertices[1];
    const HexahedronSetTopologyContainer::HexahedraAroundVertex& elemAVertexM = topoCon->getHexahedraAroundVertex(1);

    EXPECT_EQ(elemAVertex.size(), elemAVertexM.size());
    for (size_t i = 0; i < elemAVertex.size(); i++)
        EXPECT_EQ(elemAVertex[i], elemAVertexM[i]);

    // check HexahedraAroundVertex buffer element for this file    
    EXPECT_EQ(elemAVertex[0], 0);
    EXPECT_EQ(elemAVertex[1], 1);


    // Check VertexIndexInHexahedron
    const HexahedronSetTopologyContainer::Hexahedron &elem = topoCon->getHexahedron(1);
    int vId = topoCon->getVertexIndexInHexahedron(elem, elem[0]);
    EXPECT_NE(vId, -1);
    vId = topoCon->getVertexIndexInHexahedron(elem, 200000);
    EXPECT_EQ(vId, -1);

    return true;
}



bool HexahedronSetTopology_test::checkTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/nine_hexa.msh", sofa::geometry::ElementType::HEXAHEDRON);
    const HexahedronSetTopologyContainer* topoCon = dynamic_cast<HexahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

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



TEST_F(HexahedronSetTopology_test, testEmptyContainer)
{
    ASSERT_TRUE(testEmptyContainer());
}

TEST_F(HexahedronSetTopology_test, testHexahedronBuffers)
{
    ASSERT_TRUE(testHexahedronBuffers());
}

TEST_F(HexahedronSetTopology_test, testQuadBuffers)
{
    ASSERT_TRUE(testQuadBuffers());
}

TEST_F(HexahedronSetTopology_test, testEdgeBuffers)
{
    ASSERT_TRUE(testEdgeBuffers());
}

TEST_F(HexahedronSetTopology_test, testVertexBuffers)
{
    ASSERT_TRUE(testVertexBuffers());
}

TEST_F(HexahedronSetTopology_test, checkTopology)
{
    ASSERT_TRUE(checkTopology());
}


// TODO: test element on Border
// TODO: test hexahedron add/remove
// TODO: test check connectivity
