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
#include <sofa/component/topology/container/dynamic/TetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/helper/system/FileRepository.h>

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;


class TetrahedronSetTopology_test : public BaseTest
{
public:
    bool testEmptyContainer();
    bool testTetrahedronBuffers();
    bool testTriangleBuffers();
    bool testEdgeBuffers();
    bool testVertexBuffers();
    bool checkTopology();
    bool testTetrahedronGeometry();

    // ground truth from obj file;
    int nbrTetrahedron = 44;
    int nbrTriangle = 112;
    int nbrEdge = 93;
    int nbrVertex = 26;
    int elemSize = 4;
};


bool TetrahedronSetTopology_test::testEmptyContainer()
{
    const TetrahedronSetTopologyContainer::SPtr topoCon = sofa::core::objectmodel::New< TetrahedronSetTopologyContainer >();
    EXPECT_EQ(topoCon->getNbTetrahedra(), 0);
    EXPECT_EQ(topoCon->getNumberOfElements(), 0);
    EXPECT_EQ(topoCon->getNumberOfTetrahedra(), 0);
    EXPECT_EQ(topoCon->getTetrahedra().size(), 0);

    EXPECT_EQ(topoCon->getNumberOfTriangles(), 0);
    EXPECT_EQ(topoCon->getNbTriangles(), 0);
    EXPECT_EQ(topoCon->getTriangles().size(), 0);

    EXPECT_EQ(topoCon->getNumberOfEdges(), 0);
    EXPECT_EQ(topoCon->getNbEdges(), 0);
    EXPECT_EQ(topoCon->getEdges().size(), 0);

    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), 0);
    EXPECT_EQ(topoCon->getNbPoints(), 0);
    
    return true;
}


bool TetrahedronSetTopology_test::testTetrahedronBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON);
    TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // Check creation of the container
    EXPECT_EQ((topoCon->getName()), std::string("topoCon"));

    // Check topology element container buffers size
    EXPECT_EQ(topoCon->getNbTetrahedra(), nbrTetrahedron);
    EXPECT_EQ(topoCon->getNumberOfElements(), nbrTetrahedron);
    EXPECT_EQ(topoCon->getNumberOfTetrahedra(), nbrTetrahedron);
    EXPECT_EQ(topoCon->getTetrahedra().size(), nbrTetrahedron);

    // check triangles buffer has been created
    EXPECT_EQ(topoCon->getNumberOfTriangles(), nbrTriangle);
    EXPECT_EQ(topoCon->getNbTriangles(), nbrTriangle);
    EXPECT_EQ(topoCon->getTriangles().size(), nbrTriangle);

    // check edges buffer has been created
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    // The first 2 elements in this file should be :
    sofa::type::fixed_array<TetrahedronSetTopologyContainer::PointID, 4> elemTruth0(22, 2, 11, 25);
    sofa::type::fixed_array<TetrahedronSetTopologyContainer::PointID, 4> elemTruth1(22, 2, 24, 11);


    // check topology element buffer
    const sofa::type::vector<TetrahedronSetTopologyContainer::Tetrahedron>& elements = topoCon->getTetrahedronArray();
    if (elements.empty())
        return false;
    
    // check topology element 
    const TetrahedronSetTopologyContainer::Tetrahedron& elem0 = elements[0];
    EXPECT_EQ(elem0.size(), 4u);
    
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem0[i], elemTruth0[i]);
    
    // check topology element indices
    int vertexID = topoCon->getVertexIndexInTetrahedron(elem0, elemTruth0[1]);
    EXPECT_EQ(vertexID, 1);
    vertexID = topoCon->getVertexIndexInTetrahedron(elem0, elemTruth0[2]);
    EXPECT_EQ(vertexID, 2);
    vertexID = topoCon->getVertexIndexInTetrahedron(elem0, 120);
    EXPECT_EQ(vertexID, -1);


    // Check topology element buffer access    
    const TetrahedronSetTopologyContainer::Tetrahedron& elem1 = topoCon->getTetrahedron(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem1[i], elemTruth1[i]);

    const TetrahedronSetTopologyContainer::Tetrahedron& elem2 = topoCon->getTetrahedron(100000);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(elem2[i], sofa::InvalidID);


    if(scene != nullptr)
        delete scene;

    return true;
}


bool TetrahedronSetTopology_test::testTriangleBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON);
    TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check triangles
    const sofa::type::vector< TetrahedronSetTopologyContainer::TetrahedraAroundTriangle >& elemAroundTriangles = topoCon->getTetrahedraAroundTriangleArray();

    // check only the triangle buffer size: Full test on triangles are done in TriangleSetTopology_test
    EXPECT_EQ(topoCon->getNumberOfTriangles(), nbrTriangle);
    EXPECT_EQ(topoCon->getNbTriangles(), nbrTriangle);
    EXPECT_EQ(topoCon->getTriangles().size(), nbrTriangle);


    // check triangle created element
    TetrahedronSetTopologyContainer::Triangle triangle = topoCon->getTriangle(0);
    EXPECT_EQ(triangle[0], 22);
    EXPECT_EQ(triangle[1], 11);
    EXPECT_EQ(triangle[2], 2);    

    // check TetrahedraAroundTriangle buffer access
    EXPECT_EQ(elemAroundTriangles.size(), nbrTriangle);
    const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle& elemATriangle = elemAroundTriangles[0];
    const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle& elemATriangleM = topoCon->getTetrahedraAroundTriangle(0);

    EXPECT_EQ(elemATriangle.size(), elemATriangleM.size());
    for (size_t i = 0; i < elemATriangle.size(); i++)
        EXPECT_EQ(elemATriangle[i], elemATriangleM[i]);

    // check TetrahedraAroundTriangle buffer element for this file
    EXPECT_EQ(elemATriangle[0], 0);
    EXPECT_EQ(elemATriangle[1], 1);


    // check TrianglesInTetrahedron buffer acces
    const sofa::type::vector< TetrahedronSetTopologyContainer::TrianglesInTetrahedron > & triangleInTetrahedra = topoCon->getTrianglesInTetrahedronArray();
    EXPECT_EQ(triangleInTetrahedra.size(), nbrTetrahedron);

    const TetrahedronSetTopologyContainer::TrianglesInTetrahedron& triangleInElem = triangleInTetrahedra[0];
    const TetrahedronSetTopologyContainer::TrianglesInTetrahedron& triangleInElemM = topoCon->getTrianglesInTetrahedron(0);

    EXPECT_EQ(triangleInElem.size(), triangleInElemM.size());
    for (size_t i = 0; i < triangleInElem.size(); i++)
        EXPECT_EQ(triangleInElem[i], triangleInElemM[i]);

    sofa::type::fixed_array<int, 4> triangleInElemTruth(3, 2, 1, 0);
    for (size_t i = 0; i<triangleInElemTruth.size(); ++i)
        EXPECT_EQ(triangleInElem[i], triangleInElemTruth[i]);


    // Check Triangle Index in Tetrahedron
    for (size_t i = 0; i<triangleInElemTruth.size(); ++i)
        EXPECT_EQ(topoCon->getTriangleIndexInTetrahedron(triangleInElem, triangleInElemTruth[i]), i);

    int triangleId = topoCon->getTriangleIndexInTetrahedron(triangleInElem, 20000);
    EXPECT_EQ(triangleId, -1);

    // check link between TetrahedraAroundTriangle and TrianglesInTetrahedron
    for (unsigned int i = 0; i < 4; ++i)
    {
        TetrahedronSetTopologyContainer::TriangleID triangleId = i;
        const TetrahedronSetTopologyContainer::TetrahedraAroundTriangle& _elemATriangle = elemAroundTriangles[triangleId];
        for (size_t j = 0; j < _elemATriangle.size(); ++j)
        {
            TetrahedronSetTopologyContainer::TetrahedronID triId = _elemATriangle[j];
            const TetrahedronSetTopologyContainer::TrianglesInTetrahedron& _triangleInElem = triangleInTetrahedra[triId];
            bool found = false;
            for (size_t k = 0; k < _triangleInElem.size(); ++k)
            {
                if (_triangleInElem[k] == triangleId)
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


bool TetrahedronSetTopology_test::testEdgeBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON);
    TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check edges
    const sofa::type::vector< TetrahedronSetTopologyContainer::TetrahedraAroundEdge >& elemAroundEdges = topoCon->getTetrahedraAroundEdgeArray();
        
    // check only the edge buffer size: Full test on edges are done in EdgeSetTopology_test
    EXPECT_EQ(topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(topoCon->getEdges().size(), nbrEdge);

    // check edge created element
    TetrahedronSetTopologyContainer::Edge edge = topoCon->getEdge(0);
    EXPECT_EQ(edge[0], 11);
    EXPECT_EQ(edge[1], 2);


    // check TetrahedronAroundEdge buffer access
    EXPECT_EQ(elemAroundEdges.size(), nbrEdge);
    const TetrahedronSetTopologyContainer::TetrahedraAroundEdge& elemAEdge = elemAroundEdges[0];
    const TetrahedronSetTopologyContainer::TetrahedraAroundEdge& elemAEdgeM = topoCon->getTetrahedraAroundEdge(0);

    EXPECT_EQ(elemAEdge.size(), elemAEdgeM.size());
    for (size_t i = 0; i < elemAEdge.size(); i++)
        EXPECT_EQ(elemAEdge[i], elemAEdgeM[i]);

    // check TetrahedronAroundEdge buffer element for this file
    EXPECT_EQ(elemAEdge[0], 0);
    EXPECT_EQ(elemAEdge[1], 1);
    EXPECT_EQ(elemAEdge[2], 2);
    EXPECT_EQ(elemAEdge[3], 3);


    // check EdgesInTetrahedron buffer acces
    const sofa::type::vector< TetrahedronSetTopologyContainer::EdgesInTetrahedron > & edgeInTetrahedra = topoCon->getEdgesInTetrahedronArray();
    EXPECT_EQ(edgeInTetrahedra.size(), nbrTetrahedron);

    const TetrahedronSetTopologyContainer::EdgesInTetrahedron& edgeInElem = edgeInTetrahedra[2];
    const TetrahedronSetTopologyContainer::EdgesInTetrahedron& edgeInElemM = topoCon->getEdgesInTetrahedron(2);

    EXPECT_EQ(edgeInElem.size(), edgeInElemM.size());
    for (size_t i = 0; i < edgeInElem.size(); i++)
        EXPECT_EQ(edgeInElem[i], edgeInElemM[i]);
    
    sofa::type::fixed_array<int, 6> edgeInElemTruth(6, 10, 8, 9, 0, 11);
    for (size_t i = 0; i<edgeInElemTruth.size(); ++i)
        EXPECT_EQ(edgeInElem[i], edgeInElemTruth[i]);
    
    
    // Check Edge Index in Tetrahedron
    for (size_t i = 0; i<edgeInElemTruth.size(); ++i)
        EXPECT_EQ(topoCon->getEdgeIndexInTetrahedron(edgeInElem, edgeInElemTruth[i]), i);

    int edgeId = topoCon->getEdgeIndexInTetrahedron(edgeInElem, 20000);
    EXPECT_EQ(edgeId, -1);

    // check link between TetrahedraAroundEdge and EdgesInTetrahedron
    for (unsigned int i = 0; i < 4; ++i)
    {
        TetrahedronSetTopologyContainer::EdgeID edgeId = i;
        const TetrahedronSetTopologyContainer::TetrahedraAroundEdge& _elemAEdge = elemAroundEdges[edgeId];
        for (size_t j = 0; j < _elemAEdge.size(); ++j)
        {
            TetrahedronSetTopologyContainer::TetrahedronID triId = _elemAEdge[j];
            const TetrahedronSetTopologyContainer::EdgesInTetrahedron& _edgeInElem = edgeInTetrahedra[triId];
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


bool TetrahedronSetTopology_test::testVertexBuffers()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON);
    TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

    if (topoCon == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    // create and check vertex buffer
    const sofa::type::vector< TetrahedronSetTopologyContainer::TetrahedraAroundVertex >& elemAroundVertices = topoCon->getTetrahedraAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(topoCon->getNbPoints(), nbrVertex); //TODO: check why 0 and not 20

    // check TetrahedraAroundVertex buffer access
    EXPECT_EQ(elemAroundVertices.size(), nbrVertex);
    const TetrahedronSetTopologyContainer::TetrahedraAroundVertex& elemAVertex = elemAroundVertices[1];
    const TetrahedronSetTopologyContainer::TetrahedraAroundVertex& elemAVertexM = topoCon->getTetrahedraAroundVertex(1);

    EXPECT_EQ(elemAVertex.size(), elemAVertexM.size());
    for (size_t i = 0; i < elemAVertex.size(); i++)
        EXPECT_EQ(elemAVertex[i], elemAVertexM[i]);

    // check TetrahedraAroundVertex buffer element for this file    
    EXPECT_EQ(elemAVertex[0], 9);
    EXPECT_EQ(elemAVertex[1], 15);
    EXPECT_EQ(elemAVertex[2], 30);
    EXPECT_EQ(elemAVertex[3], 38);
    

    // Check VertexIndexInTetrahedron
    const TetrahedronSetTopologyContainer::Tetrahedron &elem = topoCon->getTetrahedron(1);
    int vId = topoCon->getVertexIndexInTetrahedron(elem, elem[0]);
    EXPECT_NE(vId, -1);
    vId = topoCon->getVertexIndexInTetrahedron(elem, 200000);
    EXPECT_EQ(vId, -1);

    return true;
}



bool TetrahedronSetTopology_test::checkTopology()
{
    fake_TopologyScene* scene = new fake_TopologyScene("mesh/cube_low_res.msh", sofa::geometry::ElementType::TETRAHEDRON);
    const TetrahedronSetTopologyContainer* topoCon = dynamic_cast<TetrahedronSetTopologyContainer*>(scene->getNode().get()->getMeshTopology());

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

bool TetrahedronSetTopology_test::testTetrahedronGeometry()
{
    typedef sofa::component::topology::container::dynamic::TetrahedronSetGeometryAlgorithms<sofa::defaulttype::Vec3Types> TetraAlgo3;

    fake_TopologyScene* scene = new fake_TopologyScene("mesh/6_tetra_bad.msh", sofa::geometry::ElementType::TETRAHEDRON);

    std::vector<TetraAlgo3*> algos;
    scene->getNode()->get<TetraAlgo3>(&algos, sofa::core::objectmodel::BaseContext::SearchRoot);

    if (algos.empty() || algos.size() > 1)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }

    TetraAlgo3* tetraAlgo = algos[0];
    if (tetraAlgo == nullptr)
    {
        if (scene != nullptr)
            delete scene;
        return false;
    }
    
    const sofa::type::vector<sofa::core::topology::BaseMeshTopology::TetraID>& badTetra = tetraAlgo->computeBadTetrahedron();

    EXPECT_EQ(badTetra.size(), 4);

    return true;
}



TEST_F(TetrahedronSetTopology_test, testEmptyContainer)
{
    ASSERT_TRUE(testEmptyContainer());
}

TEST_F(TetrahedronSetTopology_test, testTetrahedronBuffers)
{
    ASSERT_TRUE(testTetrahedronBuffers());
}

TEST_F(TetrahedronSetTopology_test, testTriangleBuffers)
{
    ASSERT_TRUE(testTriangleBuffers());
}

TEST_F(TetrahedronSetTopology_test, testEdgeBuffers)
{
    ASSERT_TRUE(testEdgeBuffers());
}

TEST_F(TetrahedronSetTopology_test, testVertexBuffers)
{
    ASSERT_TRUE(testVertexBuffers());
}

TEST_F(TetrahedronSetTopology_test, checkTopology)
{
    ASSERT_TRUE(checkTopology());
}

TEST_F(TetrahedronSetTopology_test, testTetrahedronGeometry)
{
    ASSERT_TRUE(testTetrahedronGeometry());
}



// TODO epernod 2018-07-05: test element on Border
// TODO epernod 2018-07-05: test hexahedron add/remove
// TODO epernod 2018-07-05: test check connectivity
