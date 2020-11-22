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

#include <SofaGeneralEngine/JoinPoints.h>
using sofa::component::engine::JoinPoints;

namespace sofa
{

using defaulttype::Vector3;

template <typename _DataTypes>
class JoinPoints_test : public ::testing::Test, public JoinPoints<_DataTypes>
{
public:
    typedef _DataTypes DataTypes;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::Real Real;
    typedef sofa::helper::Quater<SReal> Quat;

    JoinPoints_test()
    {
    }

    void testData()
    {
        EXPECT_TRUE(this->findData("points") != nullptr);
        EXPECT_TRUE(this->findData("distance") != nullptr);
        EXPECT_TRUE(this->findData("mergedPoints") != nullptr);
    }

    void testNoInput()
    {
        EXPECT_MSG_EMIT(Error);
        this->doUpdate();
    }


    void testValue(const VecCoord& inputPoints, Real inputDistance, const VecCoord& expectedPoints)
    {
        EXPECT_MSG_NOEMIT(Error);
        this->f_points.setValue(inputPoints);
        this->f_distance.setValue(inputDistance);
        
        this->doUpdate();
        helper::ReadAccessor<Data<VecCoord> > outputPoints = this->f_mergedPoints;
        ASSERT_EQ(expectedPoints.size(), outputPoints.size());

        for (size_t i = 0; i < expectedPoints.size(); i++)
        {
            EXPECT_EQ(expectedPoints[i], outputPoints[i]);
        }

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
    TYPED_TEST_CASE(JoinPoints_test, DataTypes);

    // test data setup
    TYPED_TEST(JoinPoints_test, data_setup)
    {
        this->testData();
    }

    // test no input
    TYPED_TEST(JoinPoints_test, no_input)
    {
        this->testNoInput();
    }

    // test with merge
    TYPED_TEST(JoinPoints_test, mergeCase)
    {
        typename TestFixture::VecCoord input { {0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0} };
        typename TestFixture::VecCoord expectedOutput{ {0.5, 0.5, 0.0} };

        this->testValue(input, 2.0, expectedOutput);
    }

    // test with no merge
    TYPED_TEST(JoinPoints_test, noMergeCase)
    {
        typename TestFixture::VecCoord input{ {0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0} };
        typename TestFixture::VecCoord expectedOutput{ {0.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0} };

        this->testValue(input, 0.5, expectedOutput);
    }

}// namespace

}// namespace sofa
