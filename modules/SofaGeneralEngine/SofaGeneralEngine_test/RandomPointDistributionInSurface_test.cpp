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

#include <SofaTest/Sofa_test.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaGeneralEngine/RandomPointDistributionInSurface.h>
using sofa::component::engine::RandomPointDistributionInSurface;

namespace sofa
{

using defaulttype::Vector3;

template <typename _DataTypes>
class RandomPointDistributionInSurface_test : public ::testing::Test
{
public:
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::helper::Quater<SReal> Quat;

    typedef sofa::core::topology::BaseMeshTopology::Triangle Triangle;
    typedef helper::vector<Triangle> VecTriangle;

    typename RandomPointDistributionInSurface<_DataTypes>::SPtr m_randomEngine;

    RandomPointDistributionInSurface_test()
    {
    }

    void SetUp() override 
    {
        m_randomEngine = sofa::core::objectmodel::New<RandomPointDistributionInSurface<_DataTypes> >();
    }

    void testData()
    {
        EXPECT_TRUE(m_randomEngine->findData("randomSeed") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("isVisible") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("drawOutputPoints") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("minDistanceBetweenPoints") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("numberOfInPoints") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("numberOfTests") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("vertices") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("triangles") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("inPoints") != nullptr);
        EXPECT_TRUE(m_randomEngine->findData("outPoints") != nullptr);
    }

    void testNoInput()
    {
        EXPECT_MSG_EMIT(Error);
        m_randomEngine->doUpdate();
    }

    void generate(const VecCoord& inputPoints, const VecTriangle& inputTriangles, Real minDistance, 
        unsigned int seed, unsigned int numberOfInPoints, VecCoord& outputPoints)
    {
        m_randomEngine->f_vertices.setValue(inputPoints);
        m_randomEngine->f_triangles.setValue(inputTriangles);
        m_randomEngine->randomSeed.setValue(seed);
        m_randomEngine->numberOfInPoints.setValue(numberOfInPoints);
        m_randomEngine->minDistanceBetweenPoints.setValue(minDistance);


        m_randomEngine->init();

        outputPoints = m_randomEngine->f_inPoints.getValue(); //will call doUpdate on its own
    }

};


namespace
{

// Define the list of DataTypes to instanciate
using testing::Types;
typedef Types<
    defaulttype::Vec3Types
> DataTypes; // the types to instanciate.

// Test suite for all the instanciations
TYPED_TEST_CASE(RandomPointDistributionInSurface_test, DataTypes);

// test data setup
TYPED_TEST(RandomPointDistributionInSurface_test, data_setup)
{
    this->testData();
}

//// test no input
TYPED_TEST(RandomPointDistributionInSurface_test, no_input)
{
    this->testNoInput();
}

//// test with a not closed mesh
TYPED_TEST(RandomPointDistributionInSurface_test, illFormedMesh)
{
    typename TestFixture::VecCoord vertices{ {1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {3.0, 0.0, 0.0}, {4.0, 0.0, 0.0} };
    typename TestFixture::VecTriangle triangles{ {0, 2, 3}, { 1, 3, 0}, {0, 2, 1}, {1, 2, 3} };

    typename TestFixture::VecCoord outputPoints;
    const unsigned int randomSeed = 123456789;
    const unsigned int nbPoints = 10; // just asking for 10 points, otherwise takes forever to not find correct points...
    EXPECT_MSG_EMIT(Error);
    this->generate(vertices, triangles, 0.001, randomSeed, nbPoints, outputPoints); // fixed random seed
    EXPECT_MSG_EMIT(Error);
    this->generate(vertices, triangles, 0.001, 0, nbPoints, outputPoints); // true random seed
}

// test with closed tetra
TYPED_TEST(RandomPointDistributionInSurface_test, closedMesh)
{
    typename TestFixture::VecCoord vertices{ {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {-1.0, 0.0, -1.0}, {1.0, 0.0, -1.0} };
    typename TestFixture::VecTriangle triangles{ {2, 0, 3}, { 1, 3, 0}, {0, 2, 1}, {1, 2, 3} };
    
    typename TestFixture::VecCoord outputPoints;
    const unsigned int randomSeed = 123456789;
    const unsigned int nbPoints = 10;
    EXPECT_MSG_NOEMIT(Error);
    this->generate(vertices, triangles, 0.1, randomSeed, nbPoints, outputPoints); // fixed random seed
    ASSERT_EQ(outputPoints.size(), nbPoints);

    this->generate(vertices, triangles, 0.1, 0, nbPoints, outputPoints); // true random seed
    ASSERT_EQ(outputPoints.size(), nbPoints);
}

// test with seeds
TYPED_TEST(RandomPointDistributionInSurface_test, seeds)
{
    typename TestFixture::VecCoord vertices{ {0.0, 1.0, 0.0}, {1.0, 0.0, 1.0}, {-1.0, 0.0, -1.0}, {1.0, 0.0, -1.0} };
    typename TestFixture::VecTriangle triangles{ {2, 0, 3}, { 1, 3, 0}, {0, 2, 1}, {1, 2, 3} };

    typename TestFixture::VecCoord outputPoints1;
    typename TestFixture::VecCoord outputPoints2;
    const unsigned int randomSeed1 = 123456789;
    const unsigned int randomSeed2 = 987654321;
    const unsigned int nbPoints = 100;
    EXPECT_MSG_NOEMIT(Error);
    // same seed
    this->generate(vertices, triangles, 0.1, randomSeed1, nbPoints, outputPoints1);
    this->generate(vertices, triangles, 0.1, randomSeed1, nbPoints, outputPoints2);
    ASSERT_EQ(outputPoints1.size(), nbPoints);
    ASSERT_EQ(outputPoints2.size(), nbPoints);

    for (size_t i = 0; i < outputPoints1.size(); i++)
    {
        EXPECT_EQ(outputPoints1[i], outputPoints2[i]);
    }

    // different seed
    this->generate(vertices, triangles, 0.1, randomSeed1, nbPoints, outputPoints1);
    this->generate(vertices, triangles, 0.1, randomSeed2, nbPoints, outputPoints2);
    ASSERT_EQ(outputPoints1.size(), nbPoints);
    ASSERT_EQ(outputPoints2.size(), nbPoints);

    // test if at least one is different (it could be possible that two points are similar.... but REALLY unlikely)
    bool isDifferent = false;
    for (size_t i = 0; i < outputPoints1.size(); i++)
    {
        isDifferent = isDifferent || (outputPoints1[i] != outputPoints2[i]);
    }
    EXPECT_TRUE(isDifferent);


    // true random seeds
    this->generate(vertices, triangles, 0.1, 0, nbPoints, outputPoints1);
    sofa::helper::system::thread::CTime::sleep(1.1); // wait a bit in order to change seed  
    this->generate(vertices, triangles, 0.1, 0, nbPoints, outputPoints2);
    ASSERT_EQ(outputPoints1.size(), nbPoints);
    ASSERT_EQ(outputPoints2.size(), nbPoints);

    isDifferent = false;
    for (size_t i = 0; i < outputPoints1.size(); i++)
    {
        isDifferent = isDifferent || (outputPoints1[i] != outputPoints2[i]);
    }
    EXPECT_TRUE(isDifferent);
}

}// namespace

}// namespace sofa
