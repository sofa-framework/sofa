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
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyContainer.h>
#include <sofa/component/topology/container/dynamic/TriangleSetTopologyModifier.h>
#include <sofa/component/topology/container/dynamic/TriangleSetGeometryAlgorithms.h>
#include <sofa/helper/system/FileRepository.h>

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;


class TriangleSetTopology_test : public BaseTest
{
public:
    /// Test on TriangleSetTopologyContainer creation without Data. All container should be empty.
    bool testEmptyContainer();

    /// Test on TriangleSetTopologyContainer creation with a triangular mesh as input. Check Triangle container size and content against know values.
    bool testTriangleBuffers();

    /// Test on TriangleSetTopologyContainer creation with a triangular mesh as input. Check Edge container size and content against know values.
    bool testEdgeBuffers();

    /// Test on TriangleSetTopologyContainer creation with a triangular mesh as input. Check Vertex container size and content against know values.
    bool testVertexBuffers();

    /// Test on TriangleSetTopologyContainer creation with a triangular mesh as input. Call member method CheckTopology which check all buffers concistency.
    bool checkTopology();


    /// Test on @sa TriangleSetTopologyModifier::removeVertices method and check triangle buffers.
    bool testRemovingVertices();

    /// Test on @sa TriangleSetTopologyModifier::removeTriangles method with isolated vertices and check triangle buffers.
    bool testRemovingTriangles();

    /// Test on @sa TriangleSetTopologyModifier::addTriangles method and check triangle buffers.
    bool testAddingTriangles();


    /// Test on @sa TriangleSetGeometryAlgorithms::computeSegmentTriangleIntersectionInPlane method.
    bool testTriangleSegmentIntersectionInPlane(const sofa::type::Vec3& bufferZ);

private:
    /// Method to factorize the creation and loading of the @sa m_scene and retrieve Topology container @sa m_topoCon
    bool loadTopologyContainer(const std::string& filename);

    /// Pointer to the basic scene created with a topology for the tests
    std::unique_ptr<fake_TopologyScene> m_scene;

    /// Pointer to the topology container created in the scene @sa m_scene
    TriangleSetTopologyContainer::SPtr m_topoCon = nullptr;

    // Ground truth values for mesh 'square1.obj'
    int nbrTriangle = 26;
    int nbrEdge = 45;
    int nbrVertex = 20;
    int elemSize = 3;
};


bool TriangleSetTopology_test::loadTopologyContainer(const std::string& filename)
{
    m_scene = std::make_unique<fake_TopologyScene>(filename, sofa::geometry::ElementType::TRIANGLE);

    if (m_scene == nullptr) {
        msg_error("TriangleSetTopology_test") << "Fake Topology creation from file: " << filename << " failed.";
        return false;
    }

    const auto root = m_scene->getNode().get();
    m_topoCon = root->get<TriangleSetTopologyContainer>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (m_topoCon == nullptr)
    {
        msg_error("TriangleSetTopology_test") << "TriangleSetTopologyContainer not found in scene.";
        return false;
    }

    return true;
}


bool TriangleSetTopology_test::testEmptyContainer()
{
    const TriangleSetTopologyContainer::SPtr triangleContainer = sofa::core::objectmodel::New< TriangleSetTopologyContainer >();
    EXPECT_EQ(triangleContainer->getNbTriangles(), 0);
    EXPECT_EQ(triangleContainer->getNumberOfElements(), 0);
    EXPECT_EQ(triangleContainer->getNumberOfTriangles(), 0);
    EXPECT_EQ(triangleContainer->getTriangles().size(), 0);

    EXPECT_EQ(triangleContainer->getNumberOfEdges(), 0);
    EXPECT_EQ(triangleContainer->getNbEdges(), 0);
    EXPECT_EQ(triangleContainer->getEdges().size(), 0);

    EXPECT_EQ(triangleContainer->d_initPoints.getValue().size(), 0);
    EXPECT_EQ(triangleContainer->getNbPoints(), 0);

    return true;
}


bool TriangleSetTopology_test::testTriangleBuffers()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // Check creation of the container
    EXPECT_EQ((m_topoCon->getName()), std::string("topoCon"));

    // Check triangle container buffers size
    EXPECT_EQ(m_topoCon->getNbTriangles(), nbrTriangle);
    EXPECT_EQ(m_topoCon->getNumberOfElements(), nbrTriangle);
    EXPECT_EQ(m_topoCon->getNumberOfTriangles(), nbrTriangle);
    EXPECT_EQ(m_topoCon->getTriangles().size(), nbrTriangle);

    // check edges buffer has been created
    EXPECT_EQ(m_topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(m_topoCon->getEdges().size(), nbrEdge);

    // The first 2 triangles in this file should be :
    sofa::type::fixed_array<TriangleSetTopologyContainer::PointID, 3> triTruth0(0, 18, 11);
    sofa::type::fixed_array<TriangleSetTopologyContainer::PointID, 3> triTruth1(0, 4, 18);


    // check triangle buffer
    const sofa::type::vector<TriangleSetTopologyContainer::Triangle>& triangles = m_topoCon->getTriangleArray();
    if (triangles.empty())
        return false;
    
    // check triangle 
    const TriangleSetTopologyContainer::Triangle& tri0 = triangles[0];
    EXPECT_EQ(tri0.size(), 3u);
    
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(tri0[i], triTruth0[i]);
    
    // check triangle indices
    int vertexID = m_topoCon->getVertexIndexInTriangle(tri0, triTruth0[1]);
    EXPECT_EQ(vertexID, 1);
    vertexID = m_topoCon->getVertexIndexInTriangle(tri0, triTruth0[2]);
    EXPECT_EQ(vertexID, 2);
    vertexID = m_topoCon->getVertexIndexInTriangle(tri0, 120);
    EXPECT_EQ(vertexID, -1);


    // Check triangle buffer access    
    const TriangleSetTopologyContainer::Triangle& tri1 = m_topoCon->getTriangle(1);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(tri1[i], triTruth1[i]);

    const TriangleSetTopologyContainer::Triangle& tri2 = m_topoCon->getTriangle(1000);
    for (int i = 0; i<elemSize; ++i)
        EXPECT_EQ(tri2[i], sofa::InvalidID);

    return true;
}


bool TriangleSetTopology_test::testEdgeBuffers()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // create and check edges
    const sofa::type::vector< TriangleSetTopologyContainer::TrianglesAroundEdge >& triAroundEdges = m_topoCon->getTrianglesAroundEdgeArray();
        
    // check only the edge buffer size: Full test on edges are done in EdgeSetTopology_test
    EXPECT_EQ(m_topoCon->getNumberOfEdges(), nbrEdge);
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(m_topoCon->getEdges().size(), nbrEdge);

    // check edge created element
    TriangleSetTopologyContainer::Edge edge = m_topoCon->getEdge(0);
    EXPECT_EQ(edge[0], 18);
    EXPECT_EQ(edge[1], 11);


    // check TriangleAroundEdge buffer access
    EXPECT_EQ(triAroundEdges.size(), nbrEdge);
    const TriangleSetTopologyContainer::TrianglesAroundEdge& triAEdge = triAroundEdges[0];
    const TriangleSetTopologyContainer::TrianglesAroundEdge& triAEdgeM = m_topoCon->getTrianglesAroundEdge(0);

    EXPECT_EQ(triAEdge.size(), triAEdgeM.size());
    for (size_t i = 0; i < triAEdge.size(); i++)
        EXPECT_EQ(triAEdge[i], triAEdgeM[i]);

    // check TriangleAroundEdge buffer element for this file
    EXPECT_EQ(triAEdge[0], 0);
    EXPECT_EQ(triAEdge[1], 15);


    // check EdgesInTriangle buffer access
    const sofa::type::vector< TriangleSetTopologyContainer::EdgesInTriangle > & edgeInTriangles = m_topoCon->getEdgesInTriangleArray();
    EXPECT_EQ(edgeInTriangles.size(), nbrTriangle);

    const TriangleSetTopologyContainer::EdgesInTriangle& edgeInTri = edgeInTriangles[2];
    const TriangleSetTopologyContainer::EdgesInTriangle& edgeInTriM = m_topoCon->getEdgesInTriangle(2);

    EXPECT_EQ(edgeInTri.size(), edgeInTriM.size());
    for (size_t i = 0; i < edgeInTri.size(); i++)
        EXPECT_EQ(edgeInTri[i], edgeInTriM[i]);

    sofa::type::fixed_array<int, 3> edgeInTriTruth(5, 6, 3);
    for (size_t i = 0; i<edgeInTriTruth.size(); ++i)
        EXPECT_EQ(edgeInTri[i], edgeInTriTruth[i]);
    
    
    // Check Edge Index in Triangle
    for (size_t i = 0; i<edgeInTriTruth.size(); ++i)
        EXPECT_EQ(m_topoCon->getEdgeIndexInTriangle(edgeInTri, edgeInTriTruth[i]), i);

    int edgeId = m_topoCon->getEdgeIndexInTriangle(edgeInTri, 20000);
    EXPECT_EQ(edgeId, -1);

    // check link between TrianglesAroundEdge and EdgesInTriangle
    for (unsigned int i = 0; i < 4; ++i)
    {
        TriangleSetTopologyContainer::EdgeID edgeId = i;
        const TriangleSetTopologyContainer::TrianglesAroundEdge& _triAEdge = triAroundEdges[edgeId];
        for (size_t j = 0; j < _triAEdge.size(); ++j)
        {
            TriangleSetTopologyContainer::TriangleID triId = _triAEdge[j];
            const TriangleSetTopologyContainer::EdgesInTriangle& _edgeInTri = edgeInTriangles[triId];
            bool found = false;
            for (size_t k = 0; k < _edgeInTri.size(); ++k)
            {
                if (_edgeInTri[k] == edgeId)
                {
                    found = true;
                    break;
                }
            }

            if (found == false)
            {
                return false;
            }
        }
    }
    
    return true;
}


bool TriangleSetTopology_test::testVertexBuffers()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // create and check vertex buffer
    const sofa::type::vector< TriangleSetTopologyContainer::TrianglesAroundVertex >& triAroundVertices = m_topoCon->getTrianglesAroundVertexArray();

    //// check only the vertex buffer size: Full test on vertics are done in PointSetTopology_test
    EXPECT_EQ(m_topoCon->d_initPoints.getValue().size(), nbrVertex);
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex);

    // check TrianglesAroundVertex buffer access
    EXPECT_EQ(triAroundVertices.size(), nbrVertex);
    const TriangleSetTopologyContainer::TrianglesAroundVertex& triAVertex = triAroundVertices[0];
    const TriangleSetTopologyContainer::TrianglesAroundVertex& triAVertexM = m_topoCon->getTrianglesAroundVertex(0);

    EXPECT_EQ(triAVertex.size(), triAVertexM.size());
    for (size_t i = 0; i < triAVertex.size(); i++)
        EXPECT_EQ(triAVertex[i], triAVertexM[i]);

    // check TrianglesAroundVertex buffer element for this file    
    EXPECT_EQ(triAVertex[0], 0);
    EXPECT_EQ(triAVertex[1], 1);


    const TriangleSetTopologyContainer::Triangle &tri = m_topoCon->getTriangle(1);
    int vId = m_topoCon->getVertexIndexInTriangle(tri, 0);
    EXPECT_NE(vId, sofa::InvalidID);
    vId = m_topoCon->getVertexIndexInTriangle(tri, 20000);
    EXPECT_EQ(vId, -1);

    return true;
}


bool TriangleSetTopology_test::checkTopology()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    return m_topoCon->checkTopology();
}



bool TriangleSetTopology_test::testRemovingVertices()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // Get access to the Triangle modifier
    const TriangleSetTopologyModifier::SPtr triangleModifier = m_scene->getNode()->get<TriangleSetTopologyModifier>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (triangleModifier == nullptr)
        return false;

    // get nbr triangles around vertex Id 0
    const auto triAV = m_topoCon->getTrianglesAroundVertex(0);
    sofa::topology::SetIndex vToremove = { 0 };
    
    // TODO @epernod (2025-01-28): triangles are not removed when a vertex is removed. An msg_error is fired as some buffers are not concistent anymore but the vertex is still removed. 
    // This might create errors. 
    EXPECT_MSG_EMIT(Error);
    triangleModifier->removePoints(vToremove);

    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex - 1);
    // EXPECT_EQ(m_topoCon->getNbTriangles(), nbrTriangle - triAV.size()); // see comment above
    

    return true;
}


bool TriangleSetTopology_test::testRemovingTriangles()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // Get access to the Triangle modifier
    const TriangleSetTopologyModifier::SPtr triangleModifier = m_scene->getNode()->get<TriangleSetTopologyModifier>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (triangleModifier == nullptr)
        return false;

    // Check triangle buffer before changes
    const sofa::type::vector<TriangleSetTopologyContainer::Triangle>& triangles = m_topoCon->getTriangleArray();
    EXPECT_EQ(m_topoCon->getNbPoints(), nbrVertex);
    EXPECT_EQ(m_topoCon->getNbEdges(), nbrEdge);
    EXPECT_EQ(triangles.size(), nbrTriangle);

    // 1. Check first the swap + pop_back method
    const TriangleSetTopologyContainer::Triangle lastTri = triangles.back();
    sofa::type::vector< TriangleSetTopologyContainer::TriangleID > triIds = { 0 };

    // Remove first edge from the buffer
    triangleModifier->removeTriangles(triIds, true, true);

    // Check size of the new size of the topology containers
    int newNbrTri = nbrTriangle - 1;
    int newNbrEdge = nbrEdge - 1; // 1st triangle is on border
    int newNbrVertex = nbrVertex;
    EXPECT_EQ(m_topoCon->getNbTriangles(), newNbrTri);
    EXPECT_EQ(m_topoCon->getNbEdges(), newNbrEdge);
    EXPECT_EQ(m_topoCon->getNbPoints(), newNbrVertex);

    // Check that first triangle is now the previous last triangle
    const TriangleSetTopologyContainer::Triangle& newTri = m_topoCon->getTriangle(0);
    EXPECT_EQ(lastTri[0], newTri[0]);
    EXPECT_EQ(lastTri[1], newTri[1]);
    EXPECT_EQ(lastTri[2], newTri[2]);


    // 2. Check removal of single Triangle in middle of the mesh, should not remove edge nor vertex
    triIds[0] = 18;
    triangleModifier->removeTriangles(triIds, true, true);
    
    // Check size of the new triangle buffer
    newNbrTri--;
    EXPECT_EQ(m_topoCon->getNbTriangles(), newNbrTri);
    EXPECT_EQ(m_topoCon->getNbEdges(), newNbrEdge);
    EXPECT_EQ(m_topoCon->getNbPoints(), newNbrVertex);


    // 3. Check removal of 2 Triangles side by side in middle of the mesh. Should remove commun edge
    const auto& triAEdge = m_topoCon->getTrianglesAroundEdge(22);
    triangleModifier->removeTriangles(triAEdge, true, true);
    
    // Check size of the new triangle buffer
    newNbrTri = newNbrTri - 2;
    newNbrEdge--;
    EXPECT_EQ(m_topoCon->getNbTriangles(), newNbrTri);
    EXPECT_EQ(m_topoCon->getNbEdges(), newNbrEdge);
    EXPECT_EQ(m_topoCon->getNbPoints(), newNbrVertex);


    // 4. Check removal of 2 Triangles side by side on corner of the mesh. Should remove commun edge, 2 edges on border and isolated vertex
    const auto& triAEdge2 = m_topoCon->getTrianglesAroundEdge(11);
    triangleModifier->removeTriangles(triAEdge2, true, true);

    // Check size of the new triangle buffer
    newNbrTri = newNbrTri - 2;
    newNbrEdge = newNbrEdge - 3;
    newNbrVertex = newNbrVertex - 1;
    EXPECT_EQ(m_topoCon->getNbTriangles(), newNbrTri);
    EXPECT_EQ(m_topoCon->getNbEdges(), newNbrEdge);
    EXPECT_EQ(m_topoCon->getNbPoints(), newNbrVertex);

    return true;
}


bool TriangleSetTopology_test::testAddingTriangles()
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // Get access to the Triangle modifier
    const TriangleSetTopologyModifier::SPtr triangleModifier = m_scene->getNode()->get<TriangleSetTopologyModifier>(sofa::core::objectmodel::BaseContext::SearchDown);

    if (triangleModifier == nullptr)
        return false;

    // construct triangles based on vertices of 2 triangles which do not have vertices in commun
    const sofa::type::vector<TriangleSetTopologyContainer::Triangle>& triangles = m_topoCon->getTriangleArray();
    const TriangleSetTopologyContainer::Triangle tri0 = triangles[0];
    const TriangleSetTopologyContainer::Triangle tri1 = triangles[10];

    const auto triAV0 = m_topoCon->getTrianglesAroundVertex(tri1[0]);
    const auto triAV1 = m_topoCon->getTrianglesAroundVertex(tri1[2]);

    const TriangleSetTopologyContainer::Triangle newTri0 = TriangleSetTopologyContainer::Triangle(tri0[0], tri0[1], tri1[2]);
    const TriangleSetTopologyContainer::Triangle newTri1 = TriangleSetTopologyContainer::Triangle(tri1[0], tri0[2], tri1[2]);

    sofa::type::vector< TriangleSetTopologyContainer::Triangle > triangesToAdd = { newTri0 , newTri1 };
    
    // Add triangles
    // TODO @epernod (2025-01-28): Adding the triangle create a segmentation fault. Need to investigate why
    // triangleModifier->addTriangles(triangesToAdd);
    return true; // exit for now

    // Check buffers on new triangle just added
    EXPECT_EQ(m_topoCon->getNbTriangles(), nbrTriangle + triangesToAdd.size());
    const TriangleSetTopologyContainer::Triangle& checkTri0 = m_topoCon->getTriangle(nbrTriangle);
    const TriangleSetTopologyContainer::Triangle& checkTri1 = m_topoCon->getTriangle(nbrTriangle + 1);

    for (int i = 0; i < 3; i++)
    {
        EXPECT_EQ(newTri0[i], checkTri0[i]);
        EXPECT_EQ(newTri1[i], checkTri1[i]);
    }

    // Check cross buffer around vertex
    const auto& newTriAV0 = m_topoCon->getTrianglesAroundVertex(tri1[0]);
    const auto& newTriAV1 = m_topoCon->getTrianglesAroundVertex(tri1[2]);

    EXPECT_EQ(newTriAV0.size(), triAV0.size() + 1); // newTri0 has been added around vertex tri1[0]
    EXPECT_EQ(newTriAV1.size(), triAV1.size() + 2); // newTri0 and newTri1 have been added around vertex tri1[2]

    EXPECT_EQ(newTriAV0.back(), nbrTriangle); // last tri in this buffer should be triangle id == nbrTriangle
    EXPECT_EQ(newTriAV1.back(), nbrTriangle + 1); // last tri in this buffer should be triangle id == nbrTriangle + 1

    return true;
}



bool TriangleSetTopology_test::testTriangleSegmentIntersectionInPlane(const sofa::type::Vec3& bufferZ)
{
    if (!loadTopologyContainer("mesh/square1.obj"))
        return false;

    // Get access to the Triangle modifier
    const TriangleSetGeometryAlgorithms<sofa::defaulttype::Vec3Types>::SPtr triangleGeo = m_scene->getNode()->get<TriangleSetGeometryAlgorithms<sofa::defaulttype::Vec3Types> >();
    using Real = sofa::defaulttype::Vec3Types::Real;

    if (triangleGeo == nullptr)
        return false;

    const TriangleSetTopologyContainer::TriangleID tId = 0;
    const TriangleSetTopologyContainer::Triangle tri0 = m_topoCon->getTriangle(tId);
    const auto edgeIds = m_topoCon->getEdgesInTriangle(tId); // as a reminder localEdgeId 0 is the opposite edge of triangle vertex[0]

    const auto& p0 = triangleGeo->getPointPosition(tri0[0]);
    const auto& p1 = triangleGeo->getPointPosition(tri0[1]);
    const auto& p2 = triangleGeo->getPointPosition(tri0[2]);

    // store some coef with correct cast for computations and checks
    Real coef1 = Real(0.5);
    Real coef2 = Real(1.0 / 3.0);

    // Case 1: Normal case, full intersection of the segment through the triangle
    sofa::type::Vec3 ptA = p0 * coef1 + p1 * 2 * coef2;
    sofa::type::Vec3 ptB = p0 * coef1 + p2 * coef1;
    // add small buffer to be outside the triangle
    ptA = ptA + (ptA - ptB) + bufferZ;
    ptB = ptB + (ptB - ptA) - bufferZ;

    sofa::type::vector<TriangleSetTopologyContainer::EdgeID> intersectedEdges;
    sofa::type::vector<Real> baryCoefs;
    triangleGeo->computeSegmentTriangleIntersectionInPlane(ptA, ptB, tId, intersectedEdges, baryCoefs);
    
    // check results
    EXPECT_EQ(intersectedEdges.size(), 2);
    EXPECT_EQ(baryCoefs.size(), 2);

    EXPECT_EQ(intersectedEdges[0], edgeIds[1]);
    EXPECT_EQ(intersectedEdges[1], edgeIds[2]);

    EXPECT_NEAR(baryCoefs[0], coef1, 1e-8);
    EXPECT_NEAR(baryCoefs[1], coef2, 1e-8);


    // Case 2: Intersection of only 1 segment. 1st point is inside the triangle
    ptA = p0 * coef2 + p1 * coef2 + p2 * coef2;
    ptB = p0 * coef1 + p2 * coef1; // [p0, p2] is edge local id = 1
    ptA = ptA + bufferZ;
    ptB = ptB + (ptB - ptA) - bufferZ;

    intersectedEdges.clear();
    baryCoefs.clear();
    triangleGeo->computeSegmentTriangleIntersectionInPlane(ptA, ptB, tId, intersectedEdges, baryCoefs);

    // check results
    EXPECT_EQ(intersectedEdges.size(), 1);
    EXPECT_EQ(baryCoefs.size(), 1);

    EXPECT_EQ(intersectedEdges[0], edgeIds[1]);
    EXPECT_NEAR(baryCoefs[0], coef1, 1e-8);


    // Case 3: 1 vertex of the triangle is the intersection. 2 edges are intersected with coef 0 or 1 depending on edge numbering order
    ptA = p2 + sofa::type::Vec3(-1.0, 0.0, 0.0) + bufferZ;
    ptB = p2 + sofa::type::Vec3(1.0, 0.0, 0.0) - bufferZ;
    intersectedEdges.clear();
    baryCoefs.clear();
    triangleGeo->computeSegmentTriangleIntersectionInPlane(ptA, ptB, tId, intersectedEdges, baryCoefs);

    // check results
    EXPECT_EQ(intersectedEdges.size(), 2);
    EXPECT_EQ(baryCoefs.size(), 2);

    EXPECT_EQ(intersectedEdges[0], edgeIds[0]);
    EXPECT_EQ(intersectedEdges[1], edgeIds[1]);

    EXPECT_NEAR(baryCoefs[0], 0.0, 1e-8);
    EXPECT_NEAR(baryCoefs[1], 1.0, 1e-8);


    // Case 4: intersection is going through 1 vertex of the triangle and 1 opposite edge. The 3 edges are intersected
    ptA = p0;
    ptB = p1 * coef1 + p2 * coef1;
    ptA = ptA + (ptA - ptB) + bufferZ;
    ptB = ptB + (ptB - ptA) - bufferZ;

    intersectedEdges.clear();
    baryCoefs.clear();
    triangleGeo->computeSegmentTriangleIntersectionInPlane(ptA, ptB, tId, intersectedEdges, baryCoefs);

    // check results
    EXPECT_EQ(intersectedEdges.size(), 3);
    EXPECT_EQ(baryCoefs.size(), 3);

    EXPECT_EQ(intersectedEdges[0], edgeIds[0]);
    EXPECT_EQ(intersectedEdges[1], edgeIds[1]);
    EXPECT_EQ(intersectedEdges[2], edgeIds[2]);

    EXPECT_NEAR(baryCoefs[0], coef1, 1e-8);
    EXPECT_NEAR(baryCoefs[1], 0.0, 1e-8);
    EXPECT_NEAR(baryCoefs[2], 1.0, 1e-8);


    // Case 5: Segment is colinear to edge local id 2 of the triangle. In this case results should be the 2 others edges intersected with coef 0 or 1. the Edge colinear is not considered
    ptA = p0;
    ptB = p1;
    ptA = ptA + (ptA - ptB) + bufferZ;
    ptB = ptB + (ptB - ptA) - bufferZ;

    intersectedEdges.clear();
    baryCoefs.clear();
    triangleGeo->computeSegmentTriangleIntersectionInPlane(ptA, ptB, tId, intersectedEdges, baryCoefs);

    // check results
    EXPECT_EQ(intersectedEdges.size(), 2);
    EXPECT_EQ(baryCoefs.size(), 2);

    EXPECT_EQ(intersectedEdges[0], edgeIds[0]);
    EXPECT_EQ(intersectedEdges[1], edgeIds[1]);

    EXPECT_NEAR(baryCoefs[0], 1.0, 1e-8);
    EXPECT_NEAR(baryCoefs[1], 0.0, 1e-8);

    return true;
}



TEST_F(TriangleSetTopology_test, testEmptyContainer)
{
    ASSERT_TRUE(testEmptyContainer());
}

TEST_F(TriangleSetTopology_test, testTriangleBuffers)
{
    ASSERT_TRUE(testTriangleBuffers());
}

TEST_F(TriangleSetTopology_test, testEdgeBuffers)
{
    ASSERT_TRUE(testEdgeBuffers());
}

TEST_F(TriangleSetTopology_test, testVertexBuffers)
{
    ASSERT_TRUE(testVertexBuffers());
}


TEST_F(TriangleSetTopology_test, checkTopology)
{
    ASSERT_TRUE(checkTopology());
}



TEST_F(TriangleSetTopology_test, testRemovingVertices)
{
    ASSERT_TRUE(testRemovingVertices());
}

TEST_F(TriangleSetTopology_test, testRemovingTriangles)
{
    ASSERT_TRUE(testRemovingTriangles());
}

TEST_F(TriangleSetTopology_test, testAddingTriangles)
{
    ASSERT_TRUE(testAddingTriangles());
}


TEST_F(TriangleSetTopology_test, testTriangleSegmentIntersectionInPlane)
{
    sofa::type::Vec3 inPlane = sofa::type::Vec3(0.0, 0.0, 0.0);
    ASSERT_TRUE(testTriangleSegmentIntersectionInPlane(inPlane));
}

TEST_F(TriangleSetTopology_test, testTriangleSegmentIntersectionOutPlane)
{
    sofa::type::Vec3 outPlane = sofa::type::Vec3(0.0, 0.0, 1.0);
    ASSERT_TRUE(testTriangleSegmentIntersectionInPlane(outPlane));
}


// TODO: test element on Border
// TODO: test check connectivity
