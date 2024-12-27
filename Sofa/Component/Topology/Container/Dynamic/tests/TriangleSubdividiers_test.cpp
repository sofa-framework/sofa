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
    struct triangleData
    {
        TriangleID triId = sofa::InvalidID;
        Triangle tri = Triangle(sofa::InvalidID, sofa::InvalidID, sofa::InvalidID);
        
        sofa::type::fixed_array<sofa::type::Vec3, 3> triCoords;
    };

    int createTopology();

    bool testSubdivider_1Node();
    bool testSubdivider_1Edge();
    bool testSubdivider_2Edge();
    bool testSubdivider_3Edge();
    bool testSubdivider_2Node();

    std::vector <triangleData> m_triToTest;

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

/// Will test the creation of 1 point in middle of each triangle. The ouptut should be 3 triangles each time.
bool TriangleSubdividers_test::testSubdivider_1Node()
{
    int nbrP = createTopology();

    // Test barycenter point for each specific triangle
    for (unsigned int i = 0; i < m_triToTest.size(); i++)
    {
        // Create specific subdivider for 1 Node inside a triangle
        TriangleSubdivider_1Node* subdivider0 = new TriangleSubdivider_1Node(m_triToTest[i].triId, m_triToTest[i].tri);
        sofa::type::vector<PointID> ancestors = { m_triToTest[i].tri[0], m_triToTest[i].tri[1], m_triToTest[i].tri[2] };

        // Define the point to be added as the barycenter of the triangle
        SReal tier = 1._sreal / 3._sreal;
        sofa::type::vector<SReal> coefs = { tier, tier, tier };
        sofa::type::Vec3 pG = m_triToTest[i].triCoords[0] * coefs[0] + m_triToTest[i].triCoords[1] * coefs[1] + m_triToTest[i].triCoords[2] * coefs[2];

        // Add new point to the triangle and compute the subdivision
        PointToAdd* newPoint_0 = new PointToAdd(getUniqueId(m_triToTest[i].tri[0], m_triToTest[i].tri[1]), nbrP, ancestors, coefs);
        subdivider0->addPoint(newPoint_0);
        subdivider0->subdivide(m_triToTest[i].triCoords);

        // Check the structure and position of the 3 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 3);
        
        for (int j = 0; j < 3; j++)
        {
            // each triangle is composed of old edge + new barycenter point (id == 7 here)
            EXPECT_EQ(trisToAdd[j]->m_triangle[0], m_triToTest[i].tri[j]);
            EXPECT_EQ(trisToAdd[j]->m_triangle[1], m_triToTest[i].tri[(j + 1) % 3]);
            EXPECT_EQ(trisToAdd[j]->m_triangle[2], nbrP);

            EXPECT_EQ(trisToAdd[j]->m_ancestors[0], m_triToTest[i].triId); // check ancestors and barycoefs
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_coefs[0], tier);

            // check position of the new point created
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_triCoords[2][0], pG[0]);
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_triCoords[2][1], pG[1]);
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_triCoords[2][2], pG[2]);
        }
    }
 

    return true;
}


/// Will test the creation of 1 point in middle of the first edge of the triangle. The ouptut should be 2 triangles each time.
bool TriangleSubdividers_test::testSubdivider_1Edge()
{
    int nbrP = createTopology();

    // Test barycenter point for each specific triangle
    for (const triangleData& triToTest : m_triToTest)
    {
        // Create specific subdivider for 1 Node inside an edge of the triangle
        TriangleSubdivider_1Edge* subdivider0 = new TriangleSubdivider_1Edge(triToTest.triId, triToTest.tri);
        
        // Define the point to be added in middle of the first edge of a triangle
        sofa::type::vector<PointID> ancestors = { triToTest.tri[0], triToTest.tri[1]};
        sofa::type::vector<SReal> coefs = { 0.5_sreal, 0.5_sreal };
        
        sofa::type::Vec3 pG = triToTest.triCoords[0] * coefs[0] + triToTest.triCoords[1] * coefs[1];

        // Add new point to the triangle and compute the subdivision
        PointToAdd* newPoint_0 = new PointToAdd(getUniqueId(triToTest.tri[0], triToTest.tri[1]), nbrP, ancestors, coefs);
        subdivider0->addPoint(newPoint_0);
        subdivider0->subdivide(triToTest.triCoords);

        // Check the structure and position of the 3 triangles created by the subdivision
        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 2);

        // tri 1
        EXPECT_EQ(trisToAdd[0]->m_triangle[0], triToTest.tri[0]);
        EXPECT_EQ(trisToAdd[0]->m_triangle[1], nbrP);
        EXPECT_EQ(trisToAdd[0]->m_triangle[2], triToTest.tri[2]);

        EXPECT_EQ(trisToAdd[0]->m_ancestors[0], triToTest.triId);
        EXPECT_FLOAT_EQ(trisToAdd[0]->m_coefs[0], 0.5_sreal);
        EXPECT_FLOAT_EQ(trisToAdd[0]->m_triCoords[1][0], pG[0]);
        EXPECT_FLOAT_EQ(trisToAdd[0]->m_triCoords[1][1], pG[1]);
        EXPECT_FLOAT_EQ(trisToAdd[0]->m_triCoords[1][2], pG[2]);

        // tri 2
        EXPECT_EQ(trisToAdd[1]->m_triangle[0], nbrP);
        EXPECT_EQ(trisToAdd[1]->m_triangle[1], triToTest.tri[1]);
        EXPECT_EQ(trisToAdd[1]->m_triangle[2], triToTest.tri[2]);

        EXPECT_EQ(trisToAdd[1]->m_ancestors[0], triToTest.triId);
        EXPECT_FLOAT_EQ(trisToAdd[1]->m_coefs[0], 0.5_sreal);
        EXPECT_FLOAT_EQ(trisToAdd[1]->m_triCoords[0][0], pG[0]);
        EXPECT_FLOAT_EQ(trisToAdd[1]->m_triCoords[0][1], pG[1]);
        EXPECT_FLOAT_EQ(trisToAdd[1]->m_triCoords[0][2], pG[2]);
    }

    return true;
}

bool TriangleSubdividers_test::testSubdivider_2Edge()
{
    int nbrP = createTopology();

    return true;
}

bool TriangleSubdividers_test::testSubdivider_3Edge()
{
    int nbrP = createTopology();

    return true;
}

bool TriangleSubdividers_test::testSubdivider_2Node()
{
    int nbrP = createTopology();

    return true;
}



TEST_F(TriangleSubdividers_test, testSubdivider_1Node)
{
    ASSERT_TRUE(testSubdivider_1Node());
}

TEST_F(TriangleSubdividers_test, testSubdivider_1Edge)
{
    ASSERT_TRUE(testSubdivider_1Edge());
}

TEST_F(TriangleSubdividers_test, testSubdivider_2Edge)
{
    ASSERT_TRUE(testSubdivider_2Edge());
}

TEST_F(TriangleSubdividers_test, testSubdivider_3Edge)
{
    ASSERT_TRUE(testSubdivider_3Edge());
}

TEST_F(TriangleSubdividers_test, testSubdivider_2Node)
{
    ASSERT_TRUE(testSubdivider_2Node());
}
