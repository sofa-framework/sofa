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

#include <sofa/geometry/Tetrahedron.h>

#include <sofa/type/Vec.h>

#include <gtest/gtest.h>

namespace sofa
{

TEST(GeometryTetrahedron_test, volume2_vec3f)
{
    const sofa::type::Vec3f a{ -1.f, 2.f, 0.f };
    const sofa::type::Vec3f b{ 2.f, 1.f, -3.f };
    const sofa::type::Vec3f c{ 1.f, 0.f, 1.f };
    const sofa::type::Vec3f d{ 3.f, -2.f, 3.f };

    const auto testVolume = sofa::geometry::Tetrahedron::volume(a, b, c, d);
    EXPECT_NEAR(testVolume, 2.f/3.f, 1e-5);
}

TEST(GeometryTetrahedron_test, signedVolume_positive_vec3f)
{
    // Regular tetrahedron with positive orientation
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 1.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 0.f, 1.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 0.f, 1.f };

    const auto sv = sofa::geometry::Tetrahedron::signedVolume(a, b, c, d);
    EXPECT_NEAR(sv, 1.f / 6.f, 1e-5);
}

TEST(GeometryTetrahedron_test, signedVolume_negative_vec3f)
{
    // Swapping two vertices reverses orientation
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 1.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 0.f, 1.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 0.f, 1.f };

    const auto sv = sofa::geometry::Tetrahedron::signedVolume(a, c, b, d);
    EXPECT_NEAR(sv, -1.f / 6.f, 1e-5);
}

TEST(GeometryTetrahedron_test, volume_unit_vec3f)
{
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 1.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 0.f, 1.f, 0.f };
    const sofa::type::Vec3f d{ 0.f, 0.f, 1.f };

    const auto v = sofa::geometry::Tetrahedron::volume(a, b, c, d);
    EXPECT_NEAR(v, 1.f / 6.f, 1e-5);
}

TEST(GeometryTetrahedron_test, volume_degenerate_vec3f)
{
    // Degenerate: all four points coplanar
    const sofa::type::Vec3f a{ 0.f, 0.f, 0.f };
    const sofa::type::Vec3f b{ 1.f, 0.f, 0.f };
    const sofa::type::Vec3f c{ 0.f, 1.f, 0.f };
    const sofa::type::Vec3f d{ 1.f, 1.f, 0.f };

    const auto v = sofa::geometry::Tetrahedron::volume(a, b, c, d);
    EXPECT_NEAR(v, 0.f, 1e-5);
}

TEST(GeometryTetrahedron_test, volume_scaled_vec3d)
{
    // Scaled tetrahedron (double precision)
    const sofa::type::Vec3d a{ 0., 0., 0. };
    const sofa::type::Vec3d b{ 2., 0., 0. };
    const sofa::type::Vec3d c{ 0., 2., 0. };
    const sofa::type::Vec3d d{ 0., 0., 2. };

    // Volume = (2*2*2) / 6 = 8/6 = 4/3
    const auto v = sofa::geometry::Tetrahedron::volume(a, b, c, d);
    EXPECT_NEAR(v, 4.0 / 3.0, 1e-10);
}

}// namespace sofa
