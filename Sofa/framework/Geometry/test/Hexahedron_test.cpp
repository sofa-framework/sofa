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

#include <sofa/geometry/Hexahedron.h>

#include <sofa/type/fixed_array.h>
#include <array>

#include <gtest/gtest.h>

namespace sofa
{

TEST(GeometryHexahedron_test, center_vec3f)
{
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 10.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 0.f, 10.f, 0.f };
    const sofa::type::Vec3f d{ 10.f, 10.f, 0.f };
    const sofa::type::Vec3f e{ 0.f, 0.f, 10.f };
    const sofa::type::Vec3f f{ 10.f, 0.f, 10.f };
    const sofa::type::Vec3f g{ 0.f, 10.f, 10.f };
    const sofa::type::Vec3f h{ 10.f, 10.f, 10.f };

    const sofa::type::Vec3f expectedCenter{ 5.f, 5.f, 5.f };
    const auto testCenter = sofa::geometry::Hexahedron::center(a, b, c, d, e, f ,g, h);
    EXPECT_EQ(testCenter, expectedCenter);
    EXPECT_EQ(testCenter, (h-a) * 0.5f);
}

TEST(GeometryHexahedron_test, barycentricCoefficients_vec3f)
{
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 8.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 8.f, 8.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 8.f, 0.f };
    const sofa::type::Vec3f e{ 0.f, 0.f, 8.f };
    const sofa::type::Vec3f f{ 8.f, 0.f, 8.f };
    const sofa::type::Vec3f g{ 8.f, 8.f, 8.f };
    const sofa::type::Vec3f h{ 0.f, 8.f, 8.f };

    const sofa::type::Vec3f pos0{ 4.f, 4.f, 4.f };
    auto testCoeffs = sofa::geometry::Hexahedron::barycentricCoefficients(a, b, c, d, e, f, g, h, pos0);

    sofa::type::fixed_array<float,3> expectedCoeffs{ .5f, .5f, .5f };
    // no operator == for sofa::fixed_array
    EXPECT_TRUE(testCoeffs[0] == expectedCoeffs[0] && testCoeffs[1] == expectedCoeffs[1] && testCoeffs[2] == expectedCoeffs[2]);

    const sofa::type::Vec3f pos1{ 0.f, 8.f, 0.f };
    testCoeffs = sofa::geometry::Hexahedron::barycentricCoefficients(a, b, c, d, e, f, g, h, pos1);
    expectedCoeffs = sofa::type::fixed_array<float, 3>{ 0.f, 1.f, 0.f };
    EXPECT_TRUE(testCoeffs[0] == expectedCoeffs[0] && testCoeffs[1] == expectedCoeffs[1] && testCoeffs[2] == expectedCoeffs[2]);

    const sofa::type::Vec3f pos2{ 6.f, 0.f, 2.0f };
    testCoeffs = sofa::geometry::Hexahedron::barycentricCoefficients(a, b, c, d, e, f, g, h, pos2);
    expectedCoeffs = sofa::type::fixed_array<float, 3>{ 0.75f, 0.f, 0.25f };
    EXPECT_TRUE(testCoeffs[0] == expectedCoeffs[0] && testCoeffs[1] == expectedCoeffs[1] && testCoeffs[2] == expectedCoeffs[2]);
}

TEST(GeometryHexahedron_test, squaredDistanceTo_vec3f)
{
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 10.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 10.f, 10.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 10.f, 0.f };
    const sofa::type::Vec3f e{ 0.f, 0.f, 10.f };
    const sofa::type::Vec3f f{ 10.f, 0.f, 10.f };
    const sofa::type::Vec3f g{ 10.f, 10.f, 10.f };
    const sofa::type::Vec3f h{ 0.f, 10.f, 10.f };

    const sofa::type::Vec3f pos0{ 5.f, 5.f, 5.f };
    auto testSqDistance = sofa::geometry::Hexahedron::squaredDistanceTo(a, b, c, d, e, f, g, h, pos0);

    float expectedDistance = 0.f; // inside
    EXPECT_FLOAT_EQ(testSqDistance, expectedDistance * expectedDistance);

    const sofa::type::Vec3f pos1{ 15.f, 5.f, 5.f };
    testSqDistance = sofa::geometry::Hexahedron::squaredDistanceTo(a, b, c, d, e, f, g, h, pos1);
    expectedDistance = 10.f;
    EXPECT_FLOAT_EQ(testSqDistance, expectedDistance * expectedDistance);

    const sofa::type::Vec3f pos2{ -15.f, -15.f, -15.f };
    testSqDistance = sofa::geometry::Hexahedron::squaredDistanceTo(a, b, c, d, e, f, g, h, pos2);
    expectedDistance = 34.641016f;
    EXPECT_FLOAT_EQ(testSqDistance, expectedDistance * expectedDistance);
}

TEST(GeometryHexahedron_test, getPositionFromBarycentricCoefficients_vec3f)
{
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 8.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 8.f, 8.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 8.f, 0.f };
    const sofa::type::Vec3f e{ 0.f, 0.f, 8.f };
    const sofa::type::Vec3f f{ 8.f, 0.f, 8.f };
    const sofa::type::Vec3f g{ 8.f, 8.f, 8.f };
    const sofa::type::Vec3f h{ 0.f, 8.f, 8.f };

    const sofa::type::fixed_array<SReal, 3> coeffs0{ 0.5f, 0.5f, 0.5f };
    auto testPosition = sofa::geometry::Hexahedron::getPositionFromBarycentricCoefficients(a, b, c, d, e, f, g, h, coeffs0);
    sofa::type::Vec3f expectedPosition{ 4.f, 4.f, 4.f };
    // no operator == for sofa::fixed_array
    EXPECT_TRUE(testPosition[0] == expectedPosition[0] && testPosition[1] == expectedPosition[1] && testPosition[2] == expectedPosition[2]);

    const sofa::type::fixed_array<SReal, 3> coeffs1{ 1.0f, 1.0f, 1.0f };
    testPosition = sofa::geometry::Hexahedron::getPositionFromBarycentricCoefficients(a, b, c, d, e, f, g, h, coeffs1);
    expectedPosition = sofa::type::Vec3f{ 8.f, 8.f, 8.f };
    // no operator == for sofa::fixed_array
    EXPECT_TRUE(testPosition[0] == expectedPosition[0] && testPosition[1] == expectedPosition[1] && testPosition[2] == expectedPosition[2]);

    const sofa::type::fixed_array<SReal, 3> coeffs2{ 0.0f, 0.0f, 0.0f };
    testPosition = sofa::geometry::Hexahedron::getPositionFromBarycentricCoefficients(a, b, c, d, e, f, g, h, coeffs2);
    expectedPosition = sofa::type::Vec3f{ 0.f, 0.f, 0.f };
    // no operator == for sofa::fixed_array
    EXPECT_TRUE(testPosition[0] == expectedPosition[0] && testPosition[1] == expectedPosition[1] && testPosition[2] == expectedPosition[2]);

    const sofa::type::fixed_array<SReal, 3> coeffs3{ 0.1f, 0.6f, 0.3f };
    testPosition = sofa::geometry::Hexahedron::getPositionFromBarycentricCoefficients(a, b, c, d, e, f, g, h, coeffs3);
    expectedPosition = sofa::type::Vec3f{ 0.8f, 4.8f, 2.4f };
    // no operator == for sofa::fixed_array
    EXPECT_TRUE(testPosition[0] == expectedPosition[0] && testPosition[1] == expectedPosition[1] && testPosition[2] == expectedPosition[2]);
}

TEST(GeometryHexahedron_test, cube_volume_vec3f)
{
    
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 8.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 8.f, 8.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 8.f, 0.f };
    const sofa::type::Vec3f e{ 0.f, 0.f, 8.f };
    const sofa::type::Vec3f f{ 8.f, 0.f, 8.f };
    const sofa::type::Vec3f g{ 8.f, 8.f, 8.f };
    const sofa::type::Vec3f h{ 0.f, 8.f, 8.f };

    const auto testVolume = sofa::geometry::Hexahedron::volume(a, b, c, d, e, f, g, h);
    const auto expectedVolume = 8.f * 8.f * 8.f;

    EXPECT_FLOAT_EQ(testVolume, expectedVolume);
}
TEST(GeometryHexahedron_test, rand_volume_vec3f)
{
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 7.f, 0.f, 1.f };
    const sofa::type::Vec3f c{ 8.f, 8.f, 1.f };
    const sofa::type::Vec3f d{ 0.f, 9.f, 0.f };
    const sofa::type::Vec3f e{ 0.f, 0.f, 7.f };
    const sofa::type::Vec3f f{ 8.f, 0.f, 7.f };
    const sofa::type::Vec3f g{ 9.f, 8.f, 9.f };
    const sofa::type::Vec3f h{ 0.f, 7.f, 8.f };

    const auto testVolume = sofa::geometry::Hexahedron::volume(a, b, c, d, e, f, g, h);
    const auto expectedVolume = 469.16667f;

    EXPECT_FLOAT_EQ(testVolume, expectedVolume);
}
TEST(GeometryHexahedron_test, null_volume_vec3f)
{
    // special case
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };

    const auto testVolume = sofa::geometry::Hexahedron::volume(a, a, a, a, a, a, a, a);
    const auto expectedVolume = 0.f;

    EXPECT_FLOAT_EQ(testVolume, expectedVolume);
    
}

}// namespace sofa
