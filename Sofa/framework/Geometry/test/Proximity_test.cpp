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

#include <sofa/geometry/proximity/PointTriangle.h>
#include <sofa/geometry/proximity/SegmentTriangle.h>
#include <sofa/geometry/proximity/TriangleTriangle.h>

#include <sofa/type/Vec.h>

#include <gtest/gtest.h>

namespace sofa
{

// ============================================================================
// computeClosestPointOnTriangleToPoint
// ============================================================================

TEST(GeometryProximity_test, closestPointOnTriangle_pointAboveCenter)
{
    // Triangle in XY plane
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Point above the centroid
    const sofa::type::Vec3d q{ 2. / 3., 2. / 3., 5. };
    sofa::type::Vec3d closest;

    bool res = sofa::geometry::proximity::computeClosestPointOnTriangleToPoint(t0, t1, t2, q, closest);
    EXPECT_TRUE(res);
    // Closest should be the projection onto the triangle plane: centroid
    EXPECT_NEAR(closest[0], 2. / 3., 1e-4);
    EXPECT_NEAR(closest[1], 2. / 3., 1e-4);
    EXPECT_NEAR(closest[2], 0., 1e-4);
}

TEST(GeometryProximity_test, closestPointOnTriangle_pointAboveVertex)
{
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Point directly above vertex t0
    const sofa::type::Vec3d q{ 0., 0., 3. };
    sofa::type::Vec3d closest;

    bool res = sofa::geometry::proximity::computeClosestPointOnTriangleToPoint(t0, t1, t2, q, closest);
    EXPECT_TRUE(res);
    EXPECT_NEAR(closest[0], 0., 1e-4);
    EXPECT_NEAR(closest[1], 0., 1e-4);
    EXPECT_NEAR(closest[2], 0., 1e-4);
}

TEST(GeometryProximity_test, closestPointOnTriangle_pointAboveEdge)
{
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Point above the midpoint of edge t0-t1
    const sofa::type::Vec3d q{ 1., 0., 4. };
    sofa::type::Vec3d closest;

    bool res = sofa::geometry::proximity::computeClosestPointOnTriangleToPoint(t0, t1, t2, q, closest);
    EXPECT_TRUE(res);
    EXPECT_NEAR(closest[0], 1., 1e-4);
    EXPECT_NEAR(closest[1], 0., 1e-4);
    EXPECT_NEAR(closest[2], 0., 1e-4);
}

TEST(GeometryProximity_test, closestPointOnTriangle_pointFarOutside)
{
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Point far from triangle, nearest to edge t1-t2
    const sofa::type::Vec3d q{ 3., 3., 0. };
    sofa::type::Vec3d closest;

    bool res = sofa::geometry::proximity::computeClosestPointOnTriangleToPoint(t0, t1, t2, q, closest);
    EXPECT_TRUE(res);
    // The closest point on the triangle to (3,3,0) should be on the hypotenuse
    // The hypotenuse t1-t2: parameterized as t1 + t*(t2-t1) = (2,0,0) + t*(-2,2,0)
    // Project q onto the line: t = dot(q-t1, t2-t1) / dot(t2-t1, t2-t1) = dot((1,3,0),(-2,2,0))/8 = 4/8 = 0.5
    // Closest = (2,0,0) + 0.5*(-2,2,0) = (1,1,0)
    EXPECT_NEAR(closest[0], 1., 1e-4);
    EXPECT_NEAR(closest[1], 1., 1e-4);
    EXPECT_NEAR(closest[2], 0., 1e-4);
}

TEST(GeometryProximity_test, closestPointOnTriangle_degenerateTriangle)
{
    // Degenerate triangle (all points collinear)
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 1., 0., 0. };
    const sofa::type::Vec3d t2{ 2., 0., 0. };

    const sofa::type::Vec3d q{ 1., 1., 0. };
    sofa::type::Vec3d closest;

    // LCP solver may return false for degenerate triangles
    sofa::geometry::proximity::computeClosestPointOnTriangleToPoint(t0, t1, t2, q, closest);
    // Just verify it doesn't crash; result depends on LCP solver behavior
}


// ============================================================================
// computeClosestPointsSegmentAndTriangle
// ============================================================================

TEST(GeometryProximity_test, closestPointsSegmentTriangle_intersecting)
{
    // Triangle in XY plane
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Segment piercing the triangle vertically through its centroid
    const sofa::type::Vec3d s0{ 2. / 3., 2. / 3., -1. };
    const sofa::type::Vec3d s1{ 2. / 3., 2. / 3., 1. };

    sofa::type::Vec3d closestP, closestQ;
    bool res = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t0, t1, t2, s0, s1, closestP, closestQ);
    EXPECT_TRUE(res);
    // Both closest points should be at the intersection: centroid at z=0
    EXPECT_NEAR(closestP[0], 2. / 3., 1e-4);
    EXPECT_NEAR(closestP[1], 2. / 3., 1e-4);
    EXPECT_NEAR(closestP[2], 0., 1e-4);
    EXPECT_NEAR(closestQ[0], 2. / 3., 1e-4);
    EXPECT_NEAR(closestQ[1], 2. / 3., 1e-4);
    EXPECT_NEAR(closestQ[2], 0., 1e-4);
}

TEST(GeometryProximity_test, closestPointsSegmentTriangle_parallel)
{
    // Triangle in XY plane
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Segment parallel to triangle, above midpoint of edge t0-t1
    const sofa::type::Vec3d s0{ 0., 0., 1. };
    const sofa::type::Vec3d s1{ 2., 0., 1. };

    sofa::type::Vec3d closestP, closestQ;
    bool res = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t0, t1, t2, s0, s1, closestP, closestQ);
    EXPECT_TRUE(res);
    // Closest points: on the triangle edge t0-t1 and on the segment, both should have z-distance=1
    EXPECT_NEAR(closestP[2], 0., 1e-4);
    EXPECT_NEAR(closestQ[2], 1., 1e-4);
    // x coordinates should match between the two closest points
    EXPECT_NEAR(closestP[0], closestQ[0], 1e-4);
}

TEST(GeometryProximity_test, closestPointsSegmentTriangle_endpointClosest)
{
    // Triangle in XY plane
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    // Segment above and to the side, nearest endpoint is s0
    const sofa::type::Vec3d s0{ 1., 1., 1. };
    const sofa::type::Vec3d s1{ 1., 1., 5. };

    sofa::type::Vec3d closestP, closestQ;
    bool res = sofa::geometry::proximity::computeClosestPointsSegmentAndTriangle(t0, t1, t2, s0, s1, closestP, closestQ);
    EXPECT_TRUE(res);
    // Closest on segment should be s0 (gamma=0)
    EXPECT_NEAR(closestQ[0], 1., 1e-4);
    EXPECT_NEAR(closestQ[1], 1., 1e-4);
    EXPECT_NEAR(closestQ[2], 1., 1e-4);
    // Closest on triangle should be (1,1,0) which is on the hypotenuse
    EXPECT_NEAR(closestP[2], 0., 1e-4);
}


// ============================================================================
// computeClosestPointsInTwoTriangles
// ============================================================================

TEST(GeometryProximity_test, closestPointsTwoTriangles_overlapping)
{
    // Two identical coplanar triangles
    const sofa::type::Vec3d t0{ 0., 0., 0. };
    const sofa::type::Vec3d t1{ 2., 0., 0. };
    const sofa::type::Vec3d t2{ 0., 2., 0. };

    sofa::type::Vec3d closestP, closestQ;
    bool res = sofa::geometry::proximity::computeClosestPointsInTwoTriangles(t0, t1, t2, t0, t1, t2, closestP, closestQ);
    EXPECT_TRUE(res);
    // Closest points should be identical (distance=0)
    EXPECT_NEAR((closestP - closestQ).norm(), 0., 1e-4);
}

TEST(GeometryProximity_test, closestPointsTwoTriangles_separated)
{
    // Two parallel triangles separated by distance 2 along Z
    const sofa::type::Vec3d p0{ 0., 0., 0. };
    const sofa::type::Vec3d p1{ 2., 0., 0. };
    const sofa::type::Vec3d p2{ 0., 2., 0. };

    const sofa::type::Vec3d q0{ 0., 0., 2. };
    const sofa::type::Vec3d q1{ 2., 0., 2. };
    const sofa::type::Vec3d q2{ 0., 2., 2. };

    sofa::type::Vec3d closestP, closestQ;
    bool res = sofa::geometry::proximity::computeClosestPointsInTwoTriangles(p0, p1, p2, q0, q1, q2, closestP, closestQ);
    EXPECT_TRUE(res);
    // Closest points should have z=0 for P and z=2 for Q
    EXPECT_NEAR(closestP[2], 0., 1e-4);
    EXPECT_NEAR(closestQ[2], 2., 1e-4);
    // Same x,y coordinates
    EXPECT_NEAR(closestP[0], closestQ[0], 1e-4);
    EXPECT_NEAR(closestP[1], closestQ[1], 1e-4);
}

TEST(GeometryProximity_test, closestPointsTwoTriangles_edgeToEdge)
{
    // Two triangles that share no overlap, closest approach is edge-to-edge
    const sofa::type::Vec3d p0{ 0., 0., 0. };
    const sofa::type::Vec3d p1{ 1., 0., 0. };
    const sofa::type::Vec3d p2{ 0., 1., 0. };

    // Second triangle offset in x, perpendicular in z
    const sofa::type::Vec3d q0{ 2., 0., 0. };
    const sofa::type::Vec3d q1{ 3., 0., 0. };
    const sofa::type::Vec3d q2{ 2., 0., 1. };

    sofa::type::Vec3d closestP, closestQ;
    bool res = sofa::geometry::proximity::computeClosestPointsInTwoTriangles(p0, p1, p2, q0, q1, q2, closestP, closestQ);
    EXPECT_TRUE(res);
    // Closest on P should be near vertex p1=(1,0,0), closest on Q near vertex q0=(2,0,0)
    EXPECT_NEAR(closestP[0], 1., 1e-4);
    EXPECT_NEAR(closestP[1], 0., 1e-4);
    EXPECT_NEAR(closestP[2], 0., 1e-4);
    EXPECT_NEAR(closestQ[0], 2., 1e-4);
    EXPECT_NEAR(closestQ[1], 0., 1e-4);
    EXPECT_NEAR(closestQ[2], 0., 1e-4);
}

}// namespace sofa
