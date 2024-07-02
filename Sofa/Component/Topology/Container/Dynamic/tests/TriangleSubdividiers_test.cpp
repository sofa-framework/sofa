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

    void createTopology();

    bool testSubdivider_1Node();
    bool testSubdivider_1Edge();
    bool testSubdivider_2Edge();
    bool testSubdivider_3Edge();
    bool testSubdivider_2Node();

    std::vector <triangleData> m_triToTest;

};


void TriangleSubdividers_test::createTopology()
{
    triangleData squareTri;
    squareTri.triId = 0;
    squareTri.tri = { 0, 1, 2 };
    squareTri.triCoords[0] = { 0.0, 0.0, 0.0 };
    squareTri.triCoords[1] = { 1.0, 1.0, 0.0 };
    squareTri.triCoords[2] = { 0.0, 0.0, sqrt(2) };

    triangleData equiTri;
    equiTri.triId = 1;
    equiTri.tri = { 0, 3, 4 };
    equiTri.triCoords[0] = { 0.0, 0.0, 0.0 };
    equiTri.triCoords[1] = { sqrt(2) / 2._sreal, sqrt(2) / 2._sreal, 0.0 };
    equiTri.triCoords[2] = { sqrt(2) / 4._sreal, sqrt(2) / 4._sreal, sqrt(0.75_sreal)};

    triangleData flatTri;
    flatTri.triId = 2;
    flatTri.tri = { 0, 5, 6 };
    flatTri.triCoords[0] = { 0.0, 0.0, 0.0 };
    flatTri.triCoords[1] = { 10.0, 10.0, 0.0 };
    flatTri.triCoords[2] = { 0.0, 0.0, 1.0 };

    m_triToTest.emplace_back(squareTri);
    m_triToTest.emplace_back(equiTri);
    m_triToTest.emplace_back(flatTri);
}

bool TriangleSubdividers_test::testSubdivider_1Node()
{
    createTopology();

    // Test barycenter point
    for (unsigned int i = 0; i < m_triToTest.size(); i++)
    {
        TriangleSubdivider_1Node* subdivider0 = new TriangleSubdivider_1Node(m_triToTest[i].triId, m_triToTest[i].tri);
        sofa::type::vector<PointID> ancestors = { m_triToTest[i].tri[0], m_triToTest[i].tri[1], m_triToTest[i].tri[2] };

        SReal tier = 1._sreal / 3._sreal;
        sofa::type::vector<SReal> coefs = { tier, tier, tier };

        PointToAdd* newPoint_0 = new PointToAdd(getUniqueId(m_triToTest[i].tri[0], m_triToTest[i].tri[1]), 7, ancestors, coefs);
        subdivider0->addPoint(newPoint_0);
        subdivider0->subdivide(m_triToTest[i].triCoords);

        auto trisToAdd = subdivider0->getTrianglesToAdd();
        EXPECT_EQ(trisToAdd.size(), 3);
        
        for (int j = 0; j < 3; j++)
        {
            EXPECT_EQ(trisToAdd[j]->m_triangle[0], m_triToTest[i].tri[j]);
            EXPECT_EQ(trisToAdd[j]->m_triangle[1], m_triToTest[i].tri[(j + 1) % 3]);
            EXPECT_EQ(trisToAdd[j]->m_triangle[2], 7);

            EXPECT_EQ(trisToAdd[j]->m_ancestors[0], m_triToTest[i].triId);
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_coefs[0], tier);

            sofa::type::Vec3 pG = m_triToTest[i].triCoords[0] * coefs[0] + m_triToTest[i].triCoords[1] * coefs[1] + m_triToTest[i].triCoords[2] * coefs[2];
                
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_triCoords[2][0], pG[0]);
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_triCoords[2][1], pG[1]);
            EXPECT_FLOAT_EQ(trisToAdd[j]->m_triCoords[2][2], pG[2]);
        }
    }
 

    return true;
}


bool TriangleSubdividers_test::testSubdivider_1Edge()
{
    // TODO
    return true;
}

bool TriangleSubdividers_test::testSubdivider_2Edge()
{
    // TODO
    return true;
}

bool TriangleSubdividers_test::testSubdivider_3Edge()
{
    // TODO
    return true;
}

bool TriangleSubdividers_test::testSubdivider_2Node()
{
    // TODO
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
