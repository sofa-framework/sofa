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
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/type/vector.h>
#include <sofa/component/topology/container/dynamic/TriangleSubdividers.h>

using namespace sofa::component::topology::container::dynamic;
using namespace sofa::testing;

using Triangle = sofa::core::topology::BaseMeshTopology::Triangle;
using TriangleID = sofa::core::topology::BaseMeshTopology::TriangleID;

class TriangleSubdividers_test : public BaseTest
{
public:
    /// Internal structure to store Triangle information for the tests (Ids, 3 vertices indices and their coordinates)
    struct triangleData
    {
        TriangleID triId = sofa::InvalidID;
        Triangle tri = Triangle(sofa::InvalidID, sofa::InvalidID, sofa::InvalidID);
        
        sofa::type::fixed_array<sofa::type::Vec3, 3> triCoords;
    };

    /// Will create 3 triangles as the current topology with one square triangle, one equilateral triangle and one triangle nearly flat
    int createTopology();

    /// Will test the creation of 1 point in middle of each triangle. The ouptut should be 3 triangles each time.
    bool testSubdivider_1Node_baryCenter();

    /// Will test the creation of 1 point in middle of the first edge of the triangle. The ouptut should be 2 triangles each time.
    bool testSubdivider_1Edge_baryCenter();

    /// Will test the creation of 2 points in middle of the first 2 edges of the triangle. The ouptut should be 3 triangles each time.
    bool testSubdivider_2Edge_baryCenter();
    
    /// Will test the creation of 3 points in middle of the 3 edges of the triangle. The ouptut should be 4 triangles each time.
    bool testSubdivider_3Edge_baryCenter();
    
    /// Will test the creation of 2 points, one in middle of the first edge of the triangle and one in middle of the triangle. The ouptut should be 4 triangles each time.
    bool testSubdivider_2Node_baryCenter();

    std::vector <triangleData> m_triToTest;

    bool checkTriangleToAdd(TriangleToAdd* refTri, TriangleToAdd* testTri);

};

/***
* Will create 3 triangles as the current topology with one square triangle, one equilateral triangle and one triangle nearly flat
*/
int TriangleSubdividers_test::createTopology()
{
    int nbrP = 0;
    triangleData squareTri;
    squareTri.triId = 0;
    squareTri.tri = { 0, 1, 2 };
    squareTri.triCoords[0] = { 0.0, 0.0, 0.0 };
    squareTri.triCoords[1] = { 1.0, 1.0, 0.0 };
    squareTri.triCoords[2] = { 0.0, 0.0, sqrt(2) };
    nbrP += 3;

    triangleData equiTri;
    equiTri.triId = 1;
    equiTri.tri = { 0, 3, 4 };
    equiTri.triCoords[0] = { 0.0, 0.0, 0.0 };
    equiTri.triCoords[1] = { sqrt(2) / 2._sreal, sqrt(2) / 2._sreal, 0.0 };
    equiTri.triCoords[2] = { sqrt(2) / 4._sreal, sqrt(2) / 4._sreal, sqrt(0.75_sreal)};
    nbrP += 2;

    triangleData flatTri;
    flatTri.triId = 2;
    flatTri.tri = { 0, 5, 6 };
    flatTri.triCoords[0] = { 0.0, 0.0, 0.0 };
    flatTri.triCoords[1] = { 10.0, 10.0, 0.0 };
    flatTri.triCoords[2] = { 0.0, 0.0, 1.0 };
    nbrP += 2;

    m_triToTest.emplace_back(squareTri);
    m_triToTest.emplace_back(equiTri);
    m_triToTest.emplace_back(flatTri);

    return nbrP;
}

bool TriangleSubdividers_test::checkTriangleToAdd(TriangleToAdd* refTri, TriangleToAdd* testTri)
{
    // Check triangle indices
    EXPECT_EQ(refTri->m_uniqueID, testTri->m_uniqueID);
    EXPECT_EQ(refTri->m_triangle[0], testTri->m_triangle[0]);
    EXPECT_EQ(refTri->m_triangle[1], testTri->m_triangle[1]);
    EXPECT_EQ(refTri->m_triangle[2], testTri->m_triangle[2]);

    // check ancestors
    EXPECT_EQ(refTri->m_ancestors[0], testTri->m_ancestors[0]); // check ancestors and barycoefs
    EXPECT_FLOAT_EQ(refTri->m_coefs[0], testTri->m_coefs[0]);

    // check tri coordinates
    for (int vid = 0; vid < 3; vid++)
    {
        EXPECT_FLOAT_EQ(refTri->m_triCoords[vid][0], testTri->m_triCoords[vid][0]);
        EXPECT_FLOAT_EQ(refTri->m_triCoords[vid][1], testTri->m_triCoords[vid][1]);
        EXPECT_FLOAT_EQ(refTri->m_triCoords[vid][2], testTri->m_triCoords[vid][2]);
    }

    return true;
}

/// Will test the creation of 1 point in middle of each triangle. The ouptut should be 3 triangles each time.
bool TriangleSubdividers_test::testSubdivider_1Node_baryCenter()
{
    const int nbrP = createTopology();

    // Test barycenter point for each specific triangle
    for (unsigned int i = 0; i < m_triToTest.size(); i++)
    {
        // Create specific subdivider for 1 Node inside a triangle
        std::unique_ptr<TriangleSubdivider> subdivider0 = std::make_unique<TriangleSubdivider>(m_triToTest[i].triId, m_triToTest[i].tri);
        sofa::type::vector<PointID> ancestors = { m_triToTest[i].tri[0], m_triToTest[i].tri[1], m_triToTest[i].tri[2] };

        // Define the point to be added as the barycenter of the triangle
        SReal tier = 1._sreal / 3._sreal;
        sofa::type::vector<SReal> coefs = { tier, tier, tier };
        sofa::type::Vec3 pG = m_triToTest[i].triCoords[0] * coefs[0] + m_triToTest[i].triCoords[1] * coefs[1] + m_triToTest[i].triCoords[2] * coefs[2];

        // Add new point to the triangle and compute the subdivision
        std::shared_ptr<PointToAdd> newPoint_0 = std::make_shared<PointToAdd>(getUniqueId(m_triToTest[i].tri[0], m_triToTest[i].tri[1]), nbrP, ancestors, coefs);
        subdivider0->addPoint(newPoint_0);
        subdivider0->subdivide(m_triToTest[i].triCoords);

        // Check the structure and position of the 3 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 3);

        TriangleToAdd* triRef0 = new TriangleToAdd(1000000 * m_triToTest[i].triId, Triangle(m_triToTest[i].tri[0], m_triToTest[i].tri[1], nbrP), { m_triToTest[i].triId }, { tier });
        triRef0->m_triCoords = { m_triToTest[i].triCoords[0] , m_triToTest[i].triCoords[1] ,pG };
        checkTriangleToAdd(triRef0, trisToAdd[0]);
        delete triRef0;

        TriangleToAdd* triRef1 = new TriangleToAdd(1000000 * m_triToTest[i].triId + 1, Triangle(m_triToTest[i].tri[1], m_triToTest[i].tri[2], nbrP), { m_triToTest[i].triId }, { tier });
        triRef1->m_triCoords = { m_triToTest[i].triCoords[1] , m_triToTest[i].triCoords[2] , pG };
        checkTriangleToAdd(triRef1, trisToAdd[1]);
        delete triRef1;

        TriangleToAdd* triRef2 = new TriangleToAdd(1000000 * m_triToTest[i].triId + 2, Triangle(m_triToTest[i].tri[2], m_triToTest[i].tri[0], nbrP), { m_triToTest[i].triId }, { tier });
        triRef2->m_triCoords = { m_triToTest[i].triCoords[2] , m_triToTest[i].triCoords[0] , pG };
        checkTriangleToAdd(triRef2, trisToAdd[2]);
        delete triRef2;
    }
        
    return true;
}


/// Will test the creation of 1 point in middle of the first edge of the triangle. The ouptut should be 2 triangles each time.
bool TriangleSubdividers_test::testSubdivider_1Edge_baryCenter()
{
    const int nbrP = createTopology();

    // Test barycenter point for each specific triangle
    for (const triangleData& triToTest : m_triToTest)
    {
        // Create specific subdivider for 1 Node inside an edge of the triangle
        std::unique_ptr<TriangleSubdivider> subdivider0 = std::make_unique<TriangleSubdivider>(triToTest.triId, triToTest.tri);
        
        // Define the point to be added in middle of the first edge of a triangle
        sofa::type::vector<PointID> ancestors = { triToTest.tri[0], triToTest.tri[1]};
        sofa::type::vector<SReal> coefs = { 0.5_sreal, 0.5_sreal };
        
        sofa::type::Vec3 pG = triToTest.triCoords[0] * coefs[0] + triToTest.triCoords[1] * coefs[1];

        // Add new point to the triangle and compute the subdivision
        std::shared_ptr<PointToAdd> newPoint_0 = std::make_shared<PointToAdd>(getUniqueId(triToTest.tri[0], triToTest.tri[1]), nbrP, ancestors, coefs);
        subdivider0->addPoint(newPoint_0);
        subdivider0->subdivide(triToTest.triCoords);

        // Check the structure and position of the 3 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 2);

        TriangleToAdd* triRef0 = new TriangleToAdd(1000000 * triToTest.triId, Triangle(triToTest.tri[0], nbrP, triToTest.tri[2]), { triToTest.triId }, { 0.5_sreal });
        triRef0->m_triCoords = { triToTest.triCoords[0] , pG, triToTest.triCoords[2] };
        checkTriangleToAdd(triRef0, trisToAdd[0]);
        delete triRef0;

        TriangleToAdd* triRef1 = new TriangleToAdd(1000000 * triToTest.triId + 1, Triangle(nbrP, triToTest.tri[1], triToTest.tri[2]), { triToTest.triId }, { 0.5_sreal });
        triRef1->m_triCoords = { pG, triToTest.triCoords[1] , triToTest.triCoords[2] };
        checkTriangleToAdd(triRef1, trisToAdd[1]);
        delete triRef1;
    }

    return true;
}


/// Will test the creation of 2 points in middle of the first 2 edges of the triangle. The ouptut should be 3 triangles each time.
bool TriangleSubdividers_test::testSubdivider_2Edge_baryCenter()
{
    const int nbrP = createTopology();
    sofa::type::fixed_array< bool, 3> directOriented = { true, false, true };

    // Test barycenter point for each specific triangle
    for (unsigned int i = 0; i < m_triToTest.size(); i++)
    {
        const triangleData& triToTest = m_triToTest[i];

        // Create specific subdivider for 1 Node inside an edge of the triangle
        std::unique_ptr<TriangleSubdivider> subdivider0 = std::make_unique<TriangleSubdivider>(triToTest.triId, triToTest.tri);

        // Define the points to be added in middle of the first 2 edges of a triangle
        sofa::type::vector<PointID> ancestors0 = { triToTest.tri[0], triToTest.tri[1] };
        sofa::type::vector<PointID> ancestors1 = { triToTest.tri[1], triToTest.tri[2] };
        sofa::type::vector<SReal> coefs = { 0.5_sreal, 0.5_sreal };

        sofa::type::Vec3 pG0 = triToTest.triCoords[0] * coefs[0] + triToTest.triCoords[1] * coefs[1];
        sofa::type::Vec3 pG1 = triToTest.triCoords[1] * coefs[0] + triToTest.triCoords[2] * coefs[1];

        // Add new points to the triangle and compute the subdivision
        std::shared_ptr<PointToAdd> newPoint_0 = std::make_shared<PointToAdd>(getUniqueId(ancestors0[0], ancestors0[1]), nbrP, ancestors0, coefs);
        std::shared_ptr<PointToAdd> newPoint_1 = std::make_shared<PointToAdd>(getUniqueId(ancestors1[0], ancestors1[1]), nbrP+1, ancestors1, coefs);
        subdivider0->addPoint(newPoint_0);
        subdivider0->addPoint(newPoint_1);
        subdivider0->subdivide(triToTest.triCoords);

        // Check the structure and position of the 3 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 3);

        TriangleToAdd* triRef0;
        TriangleToAdd* triRef1;
        TriangleToAdd* triRef2;
        if (directOriented[i])
        {
            triRef0 = new TriangleToAdd(1000000 * triToTest.triId, Triangle(triToTest.tri[1], nbrP + 1, nbrP), { triToTest.triId }, { 0.25_sreal });
            triRef0->m_triCoords = { triToTest.triCoords[1], pG1, pG0 };

            triRef1 = new TriangleToAdd(1000000 * triToTest.triId + 1, Triangle(nbrP + 1, triToTest.tri[2], triToTest.tri[0]), { triToTest.triId }, { 0.5_sreal });
            triRef1->m_triCoords = { pG1, triToTest.triCoords[2], triToTest.triCoords[0] };

            triRef2 = new TriangleToAdd(1000000 * triToTest.triId + 2, Triangle(triToTest.tri[0], nbrP, nbrP + 1), { triToTest.triId }, { 0.25_sreal });
            triRef2->m_triCoords = { triToTest.triCoords[0], pG0, pG1 };
        }
        else
        {
            triRef0 = new TriangleToAdd(1000000 * triToTest.triId, Triangle(triToTest.tri[1], nbrP + 1, nbrP), { triToTest.triId }, { 0.25_sreal });
            triRef0->m_triCoords = { triToTest.triCoords[1] , pG1 ,pG0 };

            triRef1 = new TriangleToAdd(1000000 * triToTest.triId + 1, Triangle(nbrP + 1, triToTest.tri[2], nbrP), { triToTest.triId }, { 0.25_sreal });
            triRef1->m_triCoords = { pG1, triToTest.triCoords[2] , pG0 };

            triRef2 = new TriangleToAdd(1000000 * triToTest.triId + 2, Triangle(triToTest.tri[2], triToTest.tri[0], nbrP), { triToTest.triId }, { 0.5_sreal });
            triRef2->m_triCoords = { triToTest.triCoords[2], triToTest.triCoords[0], pG0 };
        }
        
        checkTriangleToAdd(triRef0, trisToAdd[0]);
        delete triRef0;
        
        checkTriangleToAdd(triRef1, trisToAdd[1]);
        delete triRef1;

        checkTriangleToAdd(triRef2, trisToAdd[2]);
        delete triRef2;
    }

    return true;
}


bool TriangleSubdividers_test::testSubdivider_3Edge_baryCenter()
{
    const int nbrP = createTopology();

    // Test barycenter point for each specific triangle
    for (const triangleData& triToTest : m_triToTest)
    {
        // Create specific subdivider for 1 Node inside each edge of the triangle
        std::unique_ptr<TriangleSubdivider> subdivider0 = std::make_unique<TriangleSubdivider>(triToTest.triId, triToTest.tri);
        sofa::type::fixed_array<sofa::type::Vec3, 3> pGs;
        for (unsigned int i = 0; i < 3; i++)
        {
            sofa::type::vector<SReal> coefs;
            sofa::type::vector<PointID> ancestors;

            ancestors.push_back(triToTest.tri[i]);
            coefs.push_back(0.5_sreal);
            ancestors.push_back(triToTest.tri[(i + 1) % 3]);
            coefs.push_back(0.5_sreal);

            PointID uniqID = getUniqueId(ancestors[0], ancestors[1]);
            std::shared_ptr<PointToAdd> PTA = std::make_shared<PointToAdd>(uniqID, nbrP + i, ancestors, coefs);
            subdivider0->addPoint(PTA);
            pGs[i] = triToTest.triCoords[i] * coefs[0] + triToTest.triCoords[(i + 1) % 3] * coefs[1];
        }

        subdivider0->subdivide(triToTest.triCoords);

        // Check the structure and position of the 4 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 4);

        TriangleToAdd* triRef0 = new TriangleToAdd(1000000 * triToTest.triId, Triangle(nbrP + 2, nbrP + 1, triToTest.tri[2]), { triToTest.triId }, { 0.25_sreal });
        triRef0->m_triCoords = { pGs[2], pGs[1], triToTest.triCoords[2] };
        checkTriangleToAdd(triRef0, trisToAdd[0]);
        delete triRef0;

        TriangleToAdd* triRef1 = new TriangleToAdd(1000000 * triToTest.triId + 1, Triangle(nbrP, nbrP + 2, triToTest.tri[0]), { triToTest.triId }, { 0.25_sreal });
        triRef1->m_triCoords = { pGs[0], pGs[2], triToTest.triCoords[0] };
        checkTriangleToAdd(triRef1, trisToAdd[1]);
        delete triRef1;

        TriangleToAdd* triRef2 = new TriangleToAdd(1000000 * triToTest.triId + 2, Triangle(nbrP + 1, nbrP, triToTest.tri[1]), { triToTest.triId }, { 0.25_sreal });
        triRef2->m_triCoords = { pGs[1], pGs[0], triToTest.triCoords[1] };
        checkTriangleToAdd(triRef2, trisToAdd[2]);
        delete triRef2;

        TriangleToAdd* triRef3 = new TriangleToAdd(1000000 * triToTest.triId + 3, Triangle(nbrP + 1, nbrP + 2, nbrP), { triToTest.triId }, { 0.25_sreal });
        triRef3->m_triCoords = { pGs[1], pGs[2], pGs[0] };
        checkTriangleToAdd(triRef3, trisToAdd[3]);
        delete triRef3;
    }

    return true;
}

bool TriangleSubdividers_test::testSubdivider_2Node_baryCenter()
{
    const int nbrP = createTopology();

    // Test barycenter point for each specific triangle
    for (const triangleData& triToTest : m_triToTest)
    {
        // Create specific subdivider for 1 Node inside niddle of triang and 1 node on first edge
        std::unique_ptr<TriangleSubdivider> subdivider0 = std::make_unique<TriangleSubdivider>(triToTest.triId, triToTest.tri);
        
        // Define the points to be added in middle of the triangle
        sofa::type::vector<PointID> ancestors0 = { triToTest.tri[0], triToTest.tri[1], triToTest.tri[2] };
        SReal tier = 1._sreal / 3._sreal;
        sofa::type::vector<SReal> coefs0 = { tier, tier, tier };

        // Define the points to be added in middle of the 1st edge of the triangle
        sofa::type::vector<PointID> ancestors1 = { triToTest.tri[0], triToTest.tri[1] };
        sofa::type::vector<SReal> coefs1 = { 0.5_sreal, 0.5_sreal };

        sofa::type::Vec3 pG0 = triToTest.triCoords[0] * coefs0[0] + triToTest.triCoords[1] * coefs0[1] + triToTest.triCoords[2] * coefs0[2];
        sofa::type::Vec3 pG1 = triToTest.triCoords[0] * coefs1[0] + triToTest.triCoords[1] * coefs1[1];

        // Add new points to the triangle and compute the subdivision
        std::shared_ptr<PointToAdd> newPoint_0 = std::make_shared<PointToAdd>(getUniqueId(ancestors0[0], ancestors0[1], ancestors0[2]), nbrP, ancestors0, coefs0);
        std::shared_ptr<PointToAdd> newPoint_1 = std::make_shared<PointToAdd>(getUniqueId(ancestors1[0], ancestors1[1]), nbrP + 1, ancestors1, coefs1);
        subdivider0->addPoint(newPoint_0);
        subdivider0->addPoint(newPoint_1);
        subdivider0->subdivide(triToTest.triCoords);

        // Check the structure and position of the 4 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 4);

        TriangleToAdd* triRef0 = new TriangleToAdd(1000000 * triToTest.triId, Triangle(triToTest.tri[2], triToTest.tri[0], nbrP), { triToTest.triId }, { tier });
        triRef0->m_triCoords = { triToTest.triCoords[2], triToTest.triCoords[0], pG0 };
        checkTriangleToAdd(triRef0, trisToAdd[0]);
        delete triRef0;

        TriangleToAdd* triRef1 = new TriangleToAdd(1000000 * triToTest.triId + 1, Triangle(triToTest.tri[1], triToTest.tri[2], nbrP), { triToTest.triId }, { tier });
        triRef1->m_triCoords = { triToTest.triCoords[1], triToTest.triCoords[2], pG0};
        checkTriangleToAdd(triRef1, trisToAdd[1]);
        delete triRef1;

        TriangleToAdd* triRef2 = new TriangleToAdd(1000000 * triToTest.triId + 2, Triangle(triToTest.tri[0], nbrP + 1, nbrP), { triToTest.triId }, { tier/2 });
        triRef2->m_triCoords = { triToTest.triCoords[0], pG1, pG0 };
        checkTriangleToAdd(triRef2, trisToAdd[2]);
        delete triRef2;

        TriangleToAdd* triRef3 = new TriangleToAdd(1000000 * triToTest.triId + 3, Triangle(nbrP + 1, triToTest.tri[1], nbrP), { triToTest.triId }, { tier/2 });
        triRef3->m_triCoords = { pG1, triToTest.triCoords[1], pG0 };
        checkTriangleToAdd(triRef3, trisToAdd[3]);
        delete triRef3;
    }

    return true;
}



TEST_F(TriangleSubdividers_test, testSubdivider_1Node_baryCenter)
{
    ASSERT_TRUE(testSubdivider_1Node_baryCenter());
}

TEST_F(TriangleSubdividers_test, testSubdivider_1Edge_baryCenter)
{
    ASSERT_TRUE(testSubdivider_1Edge_baryCenter());
}

TEST_F(TriangleSubdividers_test, testSubdivider_2Edge_baryCenter)
{
    ASSERT_TRUE(testSubdivider_2Edge_baryCenter());
}

TEST_F(TriangleSubdividers_test, testSubdivider_3Edge_baryCenter)
{
    ASSERT_TRUE(testSubdivider_3Edge_baryCenter());
}

TEST_F(TriangleSubdividers_test, testSubdivider_2Node_baryCenter)
{
    ASSERT_TRUE(testSubdivider_2Node_baryCenter());
}
