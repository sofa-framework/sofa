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
#include <gtest/gtest.h>
#include <sofa/type/ClampedScalar.h>

namespace sofa
{

using namespace sofa::type;

TEST(ClampedScalarTestDouble, ConstructionWithinBounds)
{
    constexpr double MIN_VAL = -10.0;
    constexpr double MAX_VAL = 20.0;
    constexpr double INIT_VAL = 5.0;

    // Initialize using bounds and a valid value
    ClampedScalar<double> s(INIT_VAL, {MIN_VAL, MAX_VAL});

    EXPECT_DOUBLE_EQ(s.getMinBound(), MIN_VAL);
    EXPECT_DOUBLE_EQ(s.getMaxBound(), MAX_VAL);
    EXPECT_DOUBLE_EQ(s.getValue(), INIT_VAL);
}

TEST(ClampedScalarTestDouble, ConstructionClampsValueTooHigh)
{
    constexpr double MIN_VAL = 0.0;
    constexpr double MAX_VAL = 10.0;
    constexpr double OVERFLOW_VAL = 99.0;

    // Initialize with a value higher than max bound
    ClampedScalar<double> s(OVERFLOW_VAL, {MIN_VAL, MAX_VAL});

    EXPECT_DOUBLE_EQ(s.getMinBound(), MIN_VAL);
    EXPECT_DOUBLE_EQ(s.getMaxBound(), MAX_VAL);
    // The value should be clamped to the maximum bound (10.0)
    EXPECT_DOUBLE_EQ(s.getValue(), 10.0);
}

TEST(ClampedScalarTestDouble, ConstructionClampsValueTooLow)
{
    constexpr double MIN_VAL = -5.0;
    constexpr double MAX_VAL = 5.0;
    constexpr double UNDERFLOW_VAL = -50.0;

    // Initialize with a value lower than min bound
    ClampedScalar<double> s(UNDERFLOW_VAL, {MIN_VAL, MAX_VAL});

    EXPECT_DOUBLE_EQ(s.getMinBound(), MIN_VAL);
    EXPECT_DOUBLE_EQ(s.getMaxBound(), MAX_VAL);
    // The value should be clamped to the minimum bound (-5.0)
    EXPECT_DOUBLE_EQ(s.getValue(), -5.0);
}

TEST(ClampedScalarTestDouble, ConstructionHandlesReversedBounds)
{
    constexpr double MIN_VAL = 10.0;
    constexpr double MAX_VAL = 0.0;  // Input bounds reversed (10 > 0)
    constexpr double INIT_VAL = 5.0;

    // Check if the constructor correctly determines min/max regardless of input order
    ClampedScalar<double> s(INIT_VAL, {MAX_VAL, MIN_VAL});

    EXPECT_DOUBLE_EQ(s.getMinBound(), 0.0);   // Should use std::min
    EXPECT_DOUBLE_EQ(s.getMaxBound(), 10.0);  // Should use std::max
}

TEST(ClampedScalarTestDouble, SetterClampsValueTooHigh)
{
    constexpr double MIN_VAL = -5.0;
    constexpr double MAX_VAL = 15.0;

    // Start with a valid initial value
    ClampedScalar<double> s(5.0, {MIN_VAL, MAX_VAL});

    // Set a value higher than max bound (20.0 -> should become 15.0)
    s = 20.0;
    EXPECT_DOUBLE_EQ(s.getValue(), 15.0);

    // Test setting exactly to the maximum bound
    s = 15.0;
    EXPECT_DOUBLE_EQ(s.getValue(), 15.0);
}

TEST(ClampedScalarTestDouble, SetterClampsValueTooLow)
{
    constexpr double MIN_VAL = -15.0;
    constexpr double MAX_VAL = 15.0;

    // Start with a valid initial value
    ClampedScalar<double> s(5.0, {MIN_VAL, MAX_VAL});

    // Set a value lower than min bound (-20.0 -> should become -15.0)
    s = -20.0;
    EXPECT_DOUBLE_EQ(s.getValue(), -15.0);
}

TEST(ClampedScalarTestDouble, CalculateClampedPredictionHigh)
{
    constexpr double MIN_VAL = 0.0;
    constexpr double MAX_VAL = 10.0;
    // Start state doesn't matter for prediction
    ClampedScalar<double> s(5.0, {MIN_VAL, MAX_VAL});

    // Predict value far above max bound (99.0 -> should be 10.0)
    EXPECT_DOUBLE_EQ(s.calculateClamped(99.0), 10.0);

    // Predict value slightly above max bound (10.1 -> should be 10.0)
    EXPECT_DOUBLE_EQ(s.calculateClamped(10.1), 10.0);
}

TEST(ClampedScalarTestDouble, CalculateClampedPredictionLow)
{
    constexpr double MIN_VAL = -10.0;
    constexpr double MAX_VAL = 10.0;
    ClampedScalar<double> s(5.0, {MIN_VAL, MAX_VAL});

    // Predict value far below min bound (-99.0 -> should be -10.0)
    EXPECT_DOUBLE_EQ(s.calculateClamped(-99.0), -10.0);

    // Predict value slightly below min bound (-10.1 -> should be -10.0)
    EXPECT_DOUBLE_EQ(s.calculateClamped(-10.1), -10.0);
}

TEST(ClampedScalarTestDouble, StreamOperatorOverloadOutput)
{
    constexpr double MIN_VAL = -10.0;
    constexpr double MAX_VAL = 10.0;
    // Use a string stream to capture output
    std::stringstream ss;

    // Clamping doesn't affect the printed value, only getValue() matters here.
    ClampedScalar<double> s(5.0, {MIN_VAL, MAX_VAL});

    ss << s;  // Should print 5.0
    EXPECT_EQ(ss.str(), "5");

    // Test stream output after setting a clamped value (15.0 -> becomes 10.0)
    s = 15.0;  // Clamps to 10.0
    std::stringstream ss2;
    ss2 << s;
    EXPECT_EQ(ss2.str(), "10");
}

TEST(ClampedScalarTestDouble, StreamOperatorOverloadInput)
{
    constexpr double MIN_VAL = -5.0;
    constexpr double MAX_VAL = 5.0;
    // Use a string stream to simulate input data
    std::stringstream ss("2");  // Input value is 2.0

    ClampedScalar<double> s(0.0, {MIN_VAL, MAX_VAL});

    ss >> s;  // Reads 2.0 (within bounds)
    EXPECT_DOUBLE_EQ(s.getValue(), 2.0);

    // Test reading a value that is too high (15.0 -> clamps to 5.0)
    std::stringstream ss_high("15");
    ClampedScalar<double> s_high(0.0, {MIN_VAL, MAX_VAL});
    ss_high >> s_high;
    EXPECT_DOUBLE_EQ(s_high.getValue(), 5.0);

    // Test reading a value that is too low (-20.0 -> clamps to -5.0)
    std::stringstream ss_low("-20");
    ClampedScalar<double> s_low(0.0, {MIN_VAL, MAX_VAL});
    ss_low >> s_low;
    EXPECT_DOUBLE_EQ(s_low.getValue(), -5.0);
}

}  // namespace sofa
