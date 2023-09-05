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
#include <sofa/core/MatrixAccumulator.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/SPtr.h>

#include <gtest/gtest.h>
#include <sofa/testing/TestMessageHandler.h>
#include <sofa/helper/logging/Message.h>

namespace sofa
{

TEST(MatrixAccumulatorInterface, Mat33)
{
    class MatrixAccumulatorInterfaceTest : core::MatrixAccumulatorInterface
    {
    public:
        using core::MatrixAccumulatorInterface::add;

        void add(sofa::SignedIndex row, sofa::SignedIndex col, float value) override
        {
            m_tripletFloat.emplace_back(row, col, value);
        }
        void add(sofa::SignedIndex row, sofa::SignedIndex col, double value) override
        {
            m_tripletDouble.emplace_back(row, col, value);
        }

        sofa::type::vector< std::tuple<sofa::SignedIndex, sofa::SignedIndex, float> > m_tripletFloat;
        sofa::type::vector< std::tuple<sofa::SignedIndex, sofa::SignedIndex, double> > m_tripletDouble;
    };

    MatrixAccumulatorInterfaceTest accumulator;

    accumulator.add(0, 0, sofa::type::Mat3x3f::Identity());

    const sofa::type::vector< std::tuple<sofa::SignedIndex, sofa::SignedIndex, float> > expectedTriplets_f{
        {0, 0, 1.f}, {0, 1, 0.f}, {0, 2, 0.f}, {1, 0, 0.f}, {1, 1, 1.f}, {1, 2, 0.f}, {2, 0, 0.f}, {2, 1, 0.f}, {2, 2, 1.f}
    };
    EXPECT_EQ(accumulator.m_tripletFloat, expectedTriplets_f);

    accumulator.add(0, 0, sofa::type::Mat3x3d::Identity());

    const sofa::type::vector< std::tuple<sofa::SignedIndex, sofa::SignedIndex, float> > expectedTriplets_d{
            {0, 0, 1.}, {0, 1, 0.}, {0, 2, 0.}, {1, 0, 0.}, {1, 1, 1.}, {1, 2, 0.}, {2, 0, 0.}, {2, 1, 0.}, {2, 2, 1.}
    };
    EXPECT_EQ(accumulator.m_tripletFloat, expectedTriplets_d);
}

TEST(MatrixAccumulatorIndexChecker, RangeVerification)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    helper::logging::MessageDispatcher::addHandler(testing::MainGtestMessageHandler::getInstance() ) ;

    // Simple class which inherits from MatrixAccumulatorInterface, but also from MatrixAccumulatorInterface
    class MatrixAccumulatorInterfaceTest : public core::MatrixAccumulatorInterface, public sofa::core::objectmodel::BaseObject
    {
    public:
        SOFA_CLASS(MatrixAccumulatorInterfaceTest, sofa::core::objectmodel::BaseObject);
        ~MatrixAccumulatorInterfaceTest() override = default;
    };

    // Alias to decorate the previous class with an index verification
    using MatrixAccumulatorIndexCheckerTest = core::MatrixAccumulatorIndexChecker<MatrixAccumulatorInterfaceTest, core::matrixaccumulator::RangeVerification>;

    const auto accumulator = sofa::core::objectmodel::New<MatrixAccumulatorIndexCheckerTest>();

    {
        EXPECT_MSG_NOEMIT(Error);
        accumulator->add(3123, 45432, 0.);
    }

    accumulator->indexVerificationStrategy = std::make_shared<core::matrixaccumulator::RangeVerification>();

    {
        EXPECT_MSG_NOEMIT(Error);
        accumulator->add(3123, 45432, 0.);
    }

    accumulator->indexVerificationStrategy->maxColIndex = 20;

    {
        EXPECT_MSG_NOEMIT(Error);
        accumulator->add(3123, 20, 0.);
    }

    accumulator->indexVerificationStrategy->maxRowIndex = 40;

    {
        EXPECT_MSG_EMIT(Error);
        accumulator->add(3123, 21, 0.);
    }

    {
        EXPECT_MSG_EMIT(Error);
        accumulator->add(41, 19, 0.);
    }

    {
        EXPECT_MSG_EMIT(Error);
        accumulator->add(41, 21, 0.);
    }

}

}
