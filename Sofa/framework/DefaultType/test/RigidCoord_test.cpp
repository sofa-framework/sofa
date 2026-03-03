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
#include <sofa/defaulttype/RigidTypes.h>
#include <sstream>
#include <iomanip>
#include <cmath>

namespace
{

using namespace sofa::defaulttype;
using sofa::type::Vec;
using sofa::type::Quat;
using sofa::type::Mat;

using Coord3d = RigidCoord<3, double>;
using Coord2d = RigidCoord<2, double>;

constexpr double tol = 1e-10;
constexpr double trigTol = 1e-9;
constexpr float ftol = 1e-5f;

const double sq2_2 = std::sqrt(2.0) / 2.0;

// ====================================================================
// RigidCoord<3,double> Tests
// ====================================================================

TEST(RigidCoord3d, DefaultConstructor)
{
    Coord3d c;
    EXPECT_NEAR(c.getCenter()[0], 0.0, tol);
    EXPECT_NEAR(c.getCenter()[1], 0.0, tol);
    EXPECT_NEAR(c.getCenter()[2], 0.0, tol);
    EXPECT_NEAR(c.getOrientation()[0], 0.0, tol);
    EXPECT_NEAR(c.getOrientation()[1], 0.0, tol);
    EXPECT_NEAR(c.getOrientation()[2], 0.0, tol);
    EXPECT_NEAR(c.getOrientation()[3], 1.0, tol);
}

TEST(RigidCoord3d, ParameterizedConstructor)
{
    Vec<3, double> pos(1.0, 2.0, 3.0);
    Quat<double> q(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(pos, q);
    EXPECT_EQ(c.getCenter(), pos);
    EXPECT_NEAR(c.getOrientation()[0], q[0], tol);
    EXPECT_NEAR(c.getOrientation()[1], q[1], tol);
    EXPECT_NEAR(c.getOrientation()[2], q[2], tol);
    EXPECT_NEAR(c.getOrientation()[3], q[3], tol);
}

TEST(RigidCoord3d, ClearAndIdentity)
{
    Vec<3, double> pos(5.0, 6.0, 7.0);
    Quat<double> q(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(pos, q);
    c.clear();
    const auto id = RigidCoord<3, double>::identity();
    EXPECT_NEAR(c.getCenter()[0], 0.0, tol);
    EXPECT_NEAR(c.getCenter()[1], 0.0, tol);
    EXPECT_NEAR(c.getCenter()[2], 0.0, tol);
    EXPECT_NEAR(c.getOrientation()[3], 1.0, tol);
    EXPECT_EQ(c, id);
}

TEST(RigidCoord3d, SizeConstants)
{
    EXPECT_EQ(Coord3d::size(), 7u);
    EXPECT_EQ(Coord3d::total_size, 7u);
    EXPECT_EQ(Coord3d::spatial_dimensions, 3u);
}

TEST(RigidCoord3d, IndexOperator)
{
    Vec<3, double> pos(1.0, 2.0, 3.0);
    Quat<double> q(0.1, 0.2, 0.3, 0.9);
    q.normalize();
    RigidCoord<3, double> c(pos, q);
    EXPECT_NEAR(c[0], 1.0, tol);
    EXPECT_NEAR(c[1], 2.0, tol);
    EXPECT_NEAR(c[2], 3.0, tol);
    EXPECT_NEAR(c[3], c.getOrientation()[0], tol);
    EXPECT_NEAR(c[4], c.getOrientation()[1], tol);
    EXPECT_NEAR(c[5], c.getOrientation()[2], tol);
    EXPECT_NEAR(c[6], c.getOrientation()[3], tol);

    // Write through operator[]
    c[0] = 10.0;
    EXPECT_NEAR(c.getCenter()[0], 10.0, tol);
    c[4] = 0.5;
    EXPECT_NEAR(c.getOrientation()[1], 0.5, tol);
}

TEST(RigidCoord3d, EqualityOperators)
{
    RigidCoord<3, double> a(Vec<3, double>(1, 2, 3), Quat<double>(0, 0, 0, 1));
    RigidCoord<3, double> b(Vec<3, double>(1, 2, 3), Quat<double>(0, 0, 0, 1));
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 4), Quat<double>(0, 0, 0, 1));
    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
    EXPECT_FALSE(a == c);
}

TEST(RigidCoord3d, RotateIdentity)
{
    auto c = RigidCoord<3, double>::identity();
    Vec<3, double> v(1.0, 2.0, 3.0);
    auto result = c.rotate(v);
    EXPECT_NEAR(result[0], 1.0, tol);
    EXPECT_NEAR(result[1], 2.0, tol);
    EXPECT_NEAR(result[2], 3.0, tol);
}

TEST(RigidCoord3d, Rotate90Z)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(0, 0, 0), q90z);
    Vec<3, double> v(1.0, 0.0, 0.0);
    auto result = c.rotate(v);
    EXPECT_NEAR(result[0], 0.0, trigTol);
    EXPECT_NEAR(result[1], 1.0, trigTol);
    EXPECT_NEAR(result[2], 0.0, trigTol);
}

TEST(RigidCoord3d, RotateInverseRotateRoundtrip)
{
    Quat<double> q(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(0, 0, 0), q);
    Vec<3, double> v(1.0, 2.0, 3.0);
    auto roundtrip = c.inverseRotate(c.rotate(v));
    EXPECT_NEAR(roundtrip[0], v[0], trigTol);
    EXPECT_NEAR(roundtrip[1], v[1], trigTol);
    EXPECT_NEAR(roundtrip[2], v[2], trigTol);
}

TEST(RigidCoord3d, Translate)
{
    RigidCoord<3, double> c(Vec<3, double>(10, 20, 30), Quat<double>(0, 0, 0, 1));
    Vec<3, double> v(1.0, 2.0, 3.0);
    auto result = c.translate(v);
    EXPECT_NEAR(result[0], 11.0, tol);
    EXPECT_NEAR(result[1], 22.0, tol);
    EXPECT_NEAR(result[2], 33.0, tol);
}

TEST(RigidCoord3d, MultVec)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(10, 0, 0), q90z);
    Vec<3, double> v(1.0, 0.0, 0.0);
    auto result = c.mult(v);
    // rotated (1,0,0) -> (0,1,0), then + (10,0,0) = (10,1,0)
    EXPECT_NEAR(result[0], 10.0, trigTol);
    EXPECT_NEAR(result[1], 1.0, trigTol);
    EXPECT_NEAR(result[2], 0.0, trigTol);
}

TEST(RigidCoord3d, MultRigidCoord)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> a(Vec<3, double>(1, 0, 0), q90z);
    RigidCoord<3, double> b(Vec<3, double>(1, 0, 0), Quat<double>(0, 0, 0, 1));
    auto result = a.mult(b);
    // center = (1,0,0) + 90z.rotate((1,0,0)) = (1,0,0) + (0,1,0) = (1,1,0)
    EXPECT_NEAR(result.getCenter()[0], 1.0, trigTol);
    EXPECT_NEAR(result.getCenter()[1], 1.0, trigTol);
    EXPECT_NEAR(result.getCenter()[2], 0.0, trigTol);
    // orientation = 90z * identity = 90z
    EXPECT_NEAR(result.getOrientation()[0], 0.0, trigTol);
    EXPECT_NEAR(result.getOrientation()[1], 0.0, trigTol);
    EXPECT_NEAR(result.getOrientation()[2], sq2_2, trigTol);
    EXPECT_NEAR(result.getOrientation()[3], sq2_2, trigTol);
}

TEST(RigidCoord3d, MultRight)
{
    RigidCoord<3, double> a(Vec<3, double>(1, 0, 0), Quat<double>(0, 0, 0, 1));
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> b(Vec<3, double>(0, 0, 1), q90z);
    a.multRight(b);
    // center += identity.rotate((0,0,1)) = (1,0,0) + (0,0,1) = (1,0,1)
    EXPECT_NEAR(a.getCenter()[0], 1.0, trigTol);
    EXPECT_NEAR(a.getCenter()[1], 0.0, trigTol);
    EXPECT_NEAR(a.getCenter()[2], 1.0, trigTol);
    // orientation = identity * 90z = 90z
    EXPECT_NEAR(a.getOrientation()[2], sq2_2, trigTol);
    EXPECT_NEAR(a.getOrientation()[3], sq2_2, trigTol);
}

TEST(RigidCoord3d, ProjectPointIdentity)
{
    auto c = RigidCoord<3, double>::identity();
    Vec<3, double> v(1.0, 2.0, 3.0);
    auto result = c.projectPoint(v);
    EXPECT_NEAR(result[0], 1.0, tol);
    EXPECT_NEAR(result[1], 2.0, tol);
    EXPECT_NEAR(result[2], 3.0, tol);
}

TEST(RigidCoord3d, ProjectUnprojectPointRoundtrip)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(5, 10, 15), q90z);
    Vec<3, double> v(1.0, 2.0, 3.0);
    auto roundtrip = c.unprojectPoint(c.projectPoint(v));
    EXPECT_NEAR(roundtrip[0], v[0], trigTol);
    EXPECT_NEAR(roundtrip[1], v[1], trigTol);
    EXPECT_NEAR(roundtrip[2], v[2], trigTol);
}

TEST(RigidCoord3d, ProjectVectorNoTranslation)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(100, 200, 300), q90z);
    Vec<3, double> v(1.0, 0.0, 0.0);
    auto result = c.projectVector(v);
    // Rotation only, no translation
    EXPECT_NEAR(result[0], 0.0, trigTol);
    EXPECT_NEAR(result[1], 1.0, trigTol);
    EXPECT_NEAR(result[2], 0.0, trigTol);
}

TEST(RigidCoord3d, ProjectUnprojectVectorRoundtrip)
{
    Quat<double> q(sq2_2, 0.0, 0.0, sq2_2); // 90 deg around X
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 3), q);
    Vec<3, double> v(4.0, 5.0, 6.0);
    auto roundtrip = c.unprojectVector(c.projectVector(v));
    EXPECT_NEAR(roundtrip[0], v[0], trigTol);
    EXPECT_NEAR(roundtrip[1], v[1], trigTol);
    EXPECT_NEAR(roundtrip[2], v[2], trigTol);
}

TEST(RigidCoord3d, ToMatrix3x3Identity)
{
    auto c = RigidCoord<3, double>::identity();
    Mat<3, 3, double> m;
    c.toMatrix(m);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(m(i, j), (i == j) ? 1.0 : 0.0, tol);
}

TEST(RigidCoord3d, ToMatrix4x4)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 3), q90z);
    Mat<4, 4, double> m;
    c.toMatrix(m);
    // Rotation block for 90 deg Z
    EXPECT_NEAR(m(0, 0), 0.0, trigTol);
    EXPECT_NEAR(m(0, 1), -1.0, trigTol);
    EXPECT_NEAR(m(1, 0), 1.0, trigTol);
    EXPECT_NEAR(m(1, 1), 0.0, trigTol);
    EXPECT_NEAR(m(2, 2), 1.0, trigTol);
    // Translation column
    EXPECT_NEAR(m(0, 3), 1.0, tol);
    EXPECT_NEAR(m(1, 3), 2.0, tol);
    EXPECT_NEAR(m(2, 3), 3.0, tol);
    // Bottom row
    EXPECT_NEAR(m(3, 0), 0.0, tol);
    EXPECT_NEAR(m(3, 1), 0.0, tol);
    EXPECT_NEAR(m(3, 2), 0.0, tol);
    EXPECT_NEAR(m(3, 3), 1.0, tol);
}

TEST(RigidCoord3d, FromMatrixToMatrixRoundtrip)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> original(Vec<3, double>(1, 2, 3), q90z);
    Mat<4, 4, double> m;
    original.toMatrix(m);
    RigidCoord<3, double> recovered;
    recovered.fromMatrix(m);
    EXPECT_NEAR(recovered.getCenter()[0], 1.0, trigTol);
    EXPECT_NEAR(recovered.getCenter()[1], 2.0, trigTol);
    EXPECT_NEAR(recovered.getCenter()[2], 3.0, trigTol);
    EXPECT_NEAR(recovered.getOrientation()[0], 0.0, trigTol);
    EXPECT_NEAR(recovered.getOrientation()[1], 0.0, trigTol);
    EXPECT_NEAR(recovered.getOrientation()[2], sq2_2, trigTol);
    EXPECT_NEAR(recovered.getOrientation()[3], sq2_2, trigTol);
}

TEST(RigidCoord3d, OperatorPlusDeriv)
{
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 3), Quat<double>(0, 0, 0, 1));
    RigidDeriv<3, double> d(Vec<3, double>(10, 20, 30), Vec<3, double>(0, 0, 0));
    auto result = c + d;
    EXPECT_NEAR(result.getCenter()[0], 11.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 22.0, tol);
    EXPECT_NEAR(result.getCenter()[2], 33.0, tol);
}

TEST(RigidCoord3d, OperatorPlusEqualAngularVelocity)
{
    RigidCoord<3, double> c; // identity
    const double halfPi = M_PI / 2.0;
    RigidDeriv<3, double> d(Vec<3, double>(0, 0, 0), Vec<3, double>(0, 0, halfPi));
    c += d;
    // Should produce a 90 deg Z rotation quaternion: (0, 0, sqrt2/2, sqrt2/2)
    EXPECT_NEAR(c.getOrientation()[0], 0.0, trigTol);
    EXPECT_NEAR(c.getOrientation()[1], 0.0, trigTol);
    EXPECT_NEAR(c.getOrientation()[2], sq2_2, trigTol);
    EXPECT_NEAR(c.getOrientation()[3], sq2_2, trigTol);
    // Verify normalization
    double qnorm = std::sqrt(
        c.getOrientation()[0] * c.getOrientation()[0] +
        c.getOrientation()[1] * c.getOrientation()[1] +
        c.getOrientation()[2] * c.getOrientation()[2] +
        c.getOrientation()[3] * c.getOrientation()[3]);
    EXPECT_NEAR(qnorm, 1.0, tol);
}

TEST(RigidCoord3d, SubtractSelf)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> a(Vec<3, double>(1, 2, 3), q90z);
    auto result = a - a;
    EXPECT_NEAR(result.getCenter()[0], 0.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 0.0, tol);
    EXPECT_NEAR(result.getCenter()[2], 0.0, tol);
    EXPECT_NEAR(result.getOrientation()[0], 0.0, trigTol);
    EXPECT_NEAR(result.getOrientation()[1], 0.0, trigTol);
    EXPECT_NEAR(result.getOrientation()[2], 0.0, trigTol);
    EXPECT_NEAR(result.getOrientation()[3], 1.0, trigTol);
}

TEST(RigidCoord3d, PlusEqualsCoordConsistency)
{
    // operator+=(RigidCoord) must produce the same result as operator+(RigidCoord)
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    Quat<double> q90x(sq2_2, 0.0, 0.0, sq2_2);
    RigidCoord<3, double> a(Vec<3, double>(1, 2, 3), q90z);
    RigidCoord<3, double> b(Vec<3, double>(4, 5, 6), q90x);

    auto sum = a + b;

    RigidCoord<3, double> accumulated = a;
    accumulated += b;

    EXPECT_NEAR(accumulated.getCenter()[0], sum.getCenter()[0], tol);
    EXPECT_NEAR(accumulated.getCenter()[1], sum.getCenter()[1], tol);
    EXPECT_NEAR(accumulated.getCenter()[2], sum.getCenter()[2], tol);
    EXPECT_NEAR(accumulated.getOrientation()[0], sum.getOrientation()[0], tol);
    EXPECT_NEAR(accumulated.getOrientation()[1], sum.getOrientation()[1], tol);
    EXPECT_NEAR(accumulated.getOrientation()[2], sum.getOrientation()[2], tol);
    EXPECT_NEAR(accumulated.getOrientation()[3], sum.getOrientation()[3], tol);
}

TEST(RigidCoord3d, ScalarMultiply)
{
    Quat<double> q90z(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 3), q90z);
    auto result = c * 2.0;
    EXPECT_NEAR(result.getCenter()[0], 2.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 4.0, tol);
    EXPECT_NEAR(result.getCenter()[2], 6.0, tol);
    // Orientation unchanged for 3D scalar multiply
    EXPECT_NEAR(result.getOrientation()[2], sq2_2, tol);
    EXPECT_NEAR(result.getOrientation()[3], sq2_2, tol);
}

TEST(RigidCoord3d, ScalarDivide)
{
    RigidCoord<3, double> c(Vec<3, double>(4, 6, 8), Quat<double>(0, 0, sq2_2, sq2_2));
    auto result = c / 2.0;
    EXPECT_NEAR(result.getCenter()[0], 2.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 3.0, tol);
    EXPECT_NEAR(result.getCenter()[2], 4.0, tol);
    EXPECT_NEAR(result.getOrientation()[2], sq2_2, tol);
}

TEST(RigidCoord3d, DotProduct)
{
    RigidCoord<3, double> a(Vec<3, double>(1, 2, 3), Quat<double>(0.1, 0.2, 0.3, 0.9));
    RigidCoord<3, double> b(Vec<3, double>(4, 5, 6), Quat<double>(0.4, 0.5, 0.6, 0.7));
    double dot = a * b;
    double expected = 1.0*4 + 2.0*5 + 3.0*6
                    + 0.1*0.4 + 0.2*0.5 + 0.3*0.6 + 0.9*0.7;
    EXPECT_NEAR(dot, expected, tol);
}

TEST(RigidCoord3d, Norm2Identity)
{
    auto c = RigidCoord<3, double>::identity();
    EXPECT_NEAR(c.norm2(), 0.0, tol);
}

TEST(RigidCoord3d, Norm2)
{
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 3), Quat<double>(0.1, 0.2, 0.3, 0.9));
    // norm2 = center.center + qx^2+qy^2+qz^2 (not qw)
    double expected = 1.0 + 4.0 + 9.0 + 0.01 + 0.04 + 0.09;
    EXPECT_NEAR(c.norm2(), expected, tol);
}

TEST(RigidCoord3d, Norm)
{
    RigidCoord<3, double> c(Vec<3, double>(1, 2, 3), Quat<double>(0.1, 0.2, 0.3, 0.9));
    EXPECT_NEAR(c.norm(), std::sqrt(c.norm2()), tol);
}

TEST(RigidCoord3d, StreamRoundtrip)
{
    Quat<double> q(0.0, 0.0, sq2_2, sq2_2);
    RigidCoord<3, double> original(Vec<3, double>(1.5, 2.5, 3.5), q);
    std::stringstream ss;
    ss << std::setprecision(17);
    ss << original;
    RigidCoord<3, double> recovered;
    ss >> recovered;
    EXPECT_NEAR(recovered.getCenter()[0], 1.5, tol);
    EXPECT_NEAR(recovered.getCenter()[1], 2.5, tol);
    EXPECT_NEAR(recovered.getCenter()[2], 3.5, tol);
    EXPECT_NEAR(recovered.getOrientation()[0], 0.0, tol);
    EXPECT_NEAR(recovered.getOrientation()[1], 0.0, tol);
    EXPECT_NEAR(recovered.getOrientation()[2], sq2_2, tol);
    EXPECT_NEAR(recovered.getOrientation()[3], sq2_2, tol);
}

// ====================================================================
// RigidCoord<2,double> Tests
// ====================================================================

TEST(RigidCoord2d, DefaultConstructor)
{
    RigidCoord<2, double> c;
    EXPECT_NEAR(c.getCenter()[0], 0.0, tol);
    EXPECT_NEAR(c.getCenter()[1], 0.0, tol);
    EXPECT_NEAR(c.getOrientation(), 0.0, tol);
}

TEST(RigidCoord2d, ParameterizedConstructor)
{
    Vec<2, double> pos(3.0, 4.0);
    RigidCoord<2, double> c(pos, 1.5);
    EXPECT_EQ(c.getCenter(), pos);
    EXPECT_NEAR(c.getOrientation(), 1.5, tol);
}

TEST(RigidCoord2d, SizeConstants)
{
    EXPECT_EQ(Coord2d::size(), 3u);
    EXPECT_EQ(Coord2d::total_size, 3u);
    EXPECT_EQ(Coord2d::spatial_dimensions, 2u);
}

TEST(RigidCoord2d, IndexOperator)
{
    RigidCoord<2, double> c(Vec<2, double>(1.0, 2.0), 0.5);
    EXPECT_NEAR(c[0], 1.0, tol);
    EXPECT_NEAR(c[1], 2.0, tol);
    EXPECT_NEAR(c[2], 0.5, tol);
    c[0] = 10.0;
    c[2] = 1.0;
    EXPECT_NEAR(c.getCenter()[0], 10.0, tol);
    EXPECT_NEAR(c.getOrientation(), 1.0, tol);
}

TEST(RigidCoord2d, Rotate90)
{
    const double halfPi = M_PI / 2.0;
    RigidCoord<2, double> c(Vec<2, double>(0, 0), halfPi);
    Vec<2, double> v(1.0, 0.0);
    auto result = c.rotate(v);
    EXPECT_NEAR(result[0], 0.0, trigTol);
    EXPECT_NEAR(result[1], 1.0, trigTol);
}

TEST(RigidCoord2d, RotateInverseRotateRoundtrip)
{
    RigidCoord<2, double> c(Vec<2, double>(0, 0), 1.2);
    Vec<2, double> v(3.0, 4.0);
    auto roundtrip = c.inverseRotate(c.rotate(v));
    EXPECT_NEAR(roundtrip[0], v[0], trigTol);
    EXPECT_NEAR(roundtrip[1], v[1], trigTol);
}

TEST(RigidCoord2d, MultVec)
{
    const double halfPi = M_PI / 2.0;
    RigidCoord<2, double> c(Vec<2, double>(10, 0), halfPi);
    Vec<2, double> v(1.0, 0.0);
    auto result = c.mult(v);
    // rotate(1,0) by pi/2 -> (0,1), then + (10,0) = (10,1)
    EXPECT_NEAR(result[0], 10.0, trigTol);
    EXPECT_NEAR(result[1], 1.0, trigTol);
}

TEST(RigidCoord2d, ProjectUnprojectRoundtrip)
{
    RigidCoord<2, double> c(Vec<2, double>(5, 10), 0.7);
    Vec<2, double> v(1.0, 2.0);
    auto roundtrip = c.unprojectPoint(c.projectPoint(v));
    EXPECT_NEAR(roundtrip[0], v[0], trigTol);
    EXPECT_NEAR(roundtrip[1], v[1], trigTol);
}

TEST(RigidCoord2d, ToMatrixIdentity)
{
    auto c = RigidCoord<2, double>::identity();
    Mat<3, 3, double> m;
    c.toMatrix(m);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(m(i, j), (i == j) ? 1.0 : 0.0, tol);
}

TEST(RigidCoord2d, FromMatrixToMatrixRoundtrip)
{
    const double halfPi = M_PI / 2.0;
    RigidCoord<2, double> original(Vec<2, double>(1, 2), halfPi);
    Mat<3, 3, double> m;
    original.toMatrix(m);
    RigidCoord<2, double> recovered;
    recovered.fromMatrix(m);
    EXPECT_NEAR(recovered.getCenter()[0], 1.0, trigTol);
    EXPECT_NEAR(recovered.getCenter()[1], 2.0, trigTol);
    EXPECT_NEAR(recovered.getOrientation(), halfPi, trigTol);
}

TEST(RigidCoord2d, OperatorPlusDeriv)
{
    RigidCoord<2, double> c(Vec<2, double>(1, 2), 0.5);
    RigidDeriv<2, double> d(Vec<2, double>(10, 20), 0.3);
    auto result = c + d;
    EXPECT_NEAR(result.getCenter()[0], 11.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 22.0, tol);
    EXPECT_NEAR(result.getOrientation(), 0.8, tol);
}

TEST(RigidCoord2d, SubtractCoords)
{
    RigidCoord<2, double> a(Vec<2, double>(5, 7), 1.0);
    RigidCoord<2, double> b(Vec<2, double>(2, 3), 0.4);
    auto result = a - b;
    EXPECT_NEAR(result.getCenter()[0], 3.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 4.0, tol);
    EXPECT_NEAR(result.getOrientation(), 0.6, tol);
}

TEST(RigidCoord2d, ScalarMultiply)
{
    RigidCoord<2, double> c(Vec<2, double>(1, 2), 0.5);
    auto result = c * 3.0;
    // 2D scales both center AND orientation
    EXPECT_NEAR(result.getCenter()[0], 3.0, tol);
    EXPECT_NEAR(result.getCenter()[1], 6.0, tol);
    EXPECT_NEAR(result.getOrientation(), 1.5, tol);
}

TEST(RigidCoord2d, DotProduct)
{
    RigidCoord<2, double> a(Vec<2, double>(1, 2), 0.5);
    RigidCoord<2, double> b(Vec<2, double>(3, 4), 0.7);
    double dot = a * b;
    double expected = 1.0*3 + 2.0*4 + 0.5*0.7;
    EXPECT_NEAR(dot, expected, tol);
}

TEST(RigidCoord2d, UnaryNegation)
{
    RigidCoord<2, double> c(Vec<2, double>(3, 4), 1.2);
    auto neg = -c;
    EXPECT_NEAR(neg.getCenter()[0], -3.0, tol);
    EXPECT_NEAR(neg.getCenter()[1], -4.0, tol);
    EXPECT_NEAR(neg.getOrientation(), -1.2, tol);

    // c + (-c) should give zero center and zero orientation
    auto sum = c + neg;
    EXPECT_NEAR(sum.getCenter()[0], 0.0, tol);
    EXPECT_NEAR(sum.getCenter()[1], 0.0, tol);
    EXPECT_NEAR(sum.getOrientation(), 0.0, tol);
}

// ====================================================================
// Float Spot-Checks
// ====================================================================

TEST(RigidCoord3f, Rotate)
{
    const float sq2_2f = std::sqrt(2.0f) / 2.0f;
    Quat<float> q90z(0.0f, 0.0f, sq2_2f, sq2_2f);
    RigidCoord<3, float> c(Vec<3, float>(0, 0, 0), q90z);
    Vec<3, float> v(1.0f, 0.0f, 0.0f);
    auto result = c.rotate(v);
    EXPECT_NEAR(result[0], 0.0f, ftol);
    EXPECT_NEAR(result[1], 1.0f, ftol);
    EXPECT_NEAR(result[2], 0.0f, ftol);
}

TEST(RigidCoord2f, Rotate)
{
    const float halfPi = static_cast<float>(M_PI / 2.0);
    RigidCoord<2, float> c(Vec<2, float>(0, 0), halfPi);
    Vec<2, float> v(1.0f, 0.0f);
    auto result = c.rotate(v);
    EXPECT_NEAR(result[0], 0.0f, ftol);
    EXPECT_NEAR(result[1], 1.0f, ftol);
}

} // anonymous namespace
