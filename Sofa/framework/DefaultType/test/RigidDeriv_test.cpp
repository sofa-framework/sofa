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
#include <sofa/defaulttype/DataTypeInfo.h>
#include <cmath>

// ====================================================================
// Existing Name test
// ====================================================================

template<sofa::Size N, typename real>
void testName(const std::string& expectedName)
{
    using Deriv = sofa::defaulttype::RigidDeriv<N, real>;
    using Block = sofa::linearalgebra::matrix_bloc_traits<Deriv, sofa::Index >;
    EXPECT_EQ(std::string(Block::Name()), expectedName);
}

TEST(RigidDerivTest, Name)
{
    testName<3, float>("RigidDeriv3f");
    testName<3, double>("RigidDeriv3d");

    testName<2, float>("RigidDeriv2f");
    testName<2, double>("RigidDeriv2d");
}

// ====================================================================
// RigidDeriv<3,double> and RigidDeriv<2,double> Tests
// ====================================================================

namespace
{

using namespace sofa::defaulttype;
using sofa::type::Vec;

constexpr double tol = 1e-10;
constexpr float ftol = 1e-5f;

// ====================================================================
// RigidDeriv<3,double> Tests
// ====================================================================

TEST(RigidDeriv3d, DefaultConstructor)
{
    RigidDeriv<3, double> d;
    for (sofa::Size i = 0; i < 6; ++i)
        EXPECT_NEAR(d[i], 0.0, tol);
}

TEST(RigidDeriv3d, ParameterizedConstructor)
{
    Vec<3, double> lin(1, 2, 3);
    Vec<3, double> ang(4, 5, 6);
    RigidDeriv<3, double> d(lin, ang);
    EXPECT_EQ(d.getVCenter(), lin);
    EXPECT_EQ(d.getVOrientation(), ang);
}

TEST(RigidDeriv3d, FromVec6)
{
    Vec<6, double> v(1, 2, 3, 4, 5, 6);
    RigidDeriv<3, double> d(v);
    EXPECT_NEAR(d[0], 1.0, tol);
    EXPECT_NEAR(d[3], 4.0, tol);
    EXPECT_NEAR(d[5], 6.0, tol);
}

TEST(RigidDeriv3d, AccessorAliases)
{
    RigidDeriv<3, double> d(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    EXPECT_EQ(d.getVCenter(), d.getLinear());
    EXPECT_EQ(d.getVOrientation(), d.getAngular());
    auto all = d.getVAll();
    for (int i = 0; i < 6; ++i)
        EXPECT_NEAR(all[i], double(i + 1), tol);
}

TEST(RigidDeriv3d, IndexOperator)
{
    RigidDeriv<3, double> d(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    EXPECT_NEAR(d[0], 1.0, tol);
    EXPECT_NEAR(d[1], 2.0, tol);
    EXPECT_NEAR(d[2], 3.0, tol);
    EXPECT_NEAR(d[3], 4.0, tol);
    EXPECT_NEAR(d[4], 5.0, tol);
    EXPECT_NEAR(d[5], 6.0, tol);
}

TEST(RigidDeriv3d, Addition)
{
    RigidDeriv<3, double> a(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    RigidDeriv<3, double> b(Vec<3, double>(10, 20, 30), Vec<3, double>(40, 50, 60));
    auto result = a + b;
    EXPECT_NEAR(result[0], 11.0, tol);
    EXPECT_NEAR(result[1], 22.0, tol);
    EXPECT_NEAR(result[2], 33.0, tol);
    EXPECT_NEAR(result[3], 44.0, tol);
    EXPECT_NEAR(result[4], 55.0, tol);
    EXPECT_NEAR(result[5], 66.0, tol);
}

TEST(RigidDeriv3d, Subtraction)
{
    RigidDeriv<3, double> a(Vec<3, double>(10, 20, 30), Vec<3, double>(40, 50, 60));
    RigidDeriv<3, double> b(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    auto result = a - b;
    EXPECT_NEAR(result[0], 9.0, tol);
    EXPECT_NEAR(result[1], 18.0, tol);
    EXPECT_NEAR(result[2], 27.0, tol);
    EXPECT_NEAR(result[3], 36.0, tol);
    EXPECT_NEAR(result[4], 45.0, tol);
    EXPECT_NEAR(result[5], 54.0, tol);
}

TEST(RigidDeriv3d, PlusEqualsMinusEquals)
{
    RigidDeriv<3, double> a(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    RigidDeriv<3, double> b(Vec<3, double>(10, 10, 10), Vec<3, double>(10, 10, 10));
    a += b;
    EXPECT_NEAR(a[0], 11.0, tol);
    EXPECT_NEAR(a[3], 14.0, tol);
    a -= b;
    EXPECT_NEAR(a[0], 1.0, tol);
    EXPECT_NEAR(a[3], 4.0, tol);
}

TEST(RigidDeriv3d, UnaryNegation)
{
    RigidDeriv<3, double> a(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    auto neg = -a;
    EXPECT_NEAR(neg[0], -1.0, tol);
    EXPECT_NEAR(neg[1], -2.0, tol);
    EXPECT_NEAR(neg[2], -3.0, tol);
    EXPECT_NEAR(neg[3], -4.0, tol);
    EXPECT_NEAR(neg[4], -5.0, tol);
    EXPECT_NEAR(neg[5], -6.0, tol);
}

TEST(RigidDeriv3d, ScalarMultiplyDivide)
{
    RigidDeriv<3, double> d(Vec<3, double>(2, 4, 6), Vec<3, double>(8, 10, 12));
    auto scaled = d * 3.0;
    EXPECT_NEAR(scaled[0], 6.0, tol);
    EXPECT_NEAR(scaled[1], 12.0, tol);
    EXPECT_NEAR(scaled[2], 18.0, tol);
    EXPECT_NEAR(scaled[3], 24.0, tol);
    EXPECT_NEAR(scaled[4], 30.0, tol);
    EXPECT_NEAR(scaled[5], 36.0, tol);

    auto divided = d / 2.0;
    EXPECT_NEAR(divided[0], 1.0, tol);
    EXPECT_NEAR(divided[1], 2.0, tol);
    EXPECT_NEAR(divided[2], 3.0, tol);
    EXPECT_NEAR(divided[3], 4.0, tol);
    EXPECT_NEAR(divided[4], 5.0, tol);
    EXPECT_NEAR(divided[5], 6.0, tol);
}

TEST(RigidDeriv3d, DotProduct)
{
    RigidDeriv<3, double> a(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    RigidDeriv<3, double> b(Vec<3, double>(7, 8, 9), Vec<3, double>(10, 11, 12));
    double dot = a * b;
    double expected = 1.0*7 + 2.0*8 + 3.0*9 + 4.0*10 + 5.0*11 + 6.0*12;
    EXPECT_NEAR(dot, expected, tol);
}

TEST(RigidDeriv3d, NormZero)
{
    RigidDeriv<3, double> d;
    EXPECT_NEAR(d.norm(), 0.0, tol);
}

TEST(RigidDeriv3d, Norm)
{
    RigidDeriv<3, double> d(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    double expected = std::sqrt(1.0 + 4.0 + 9.0 + 16.0 + 25.0 + 36.0);
    EXPECT_NEAR(d.norm(), expected, tol);
}

TEST(RigidDeriv3d, VelocityAtRotatedPointZeroAngular)
{
    RigidDeriv<3, double> d(Vec<3, double>(1, 2, 3), Vec<3, double>(0, 0, 0));
    Vec<3, double> p(10, 20, 30);
    auto result = d.velocityAtRotatedPoint(p);
    EXPECT_NEAR(result[0], 1.0, tol);
    EXPECT_NEAR(result[1], 2.0, tol);
    EXPECT_NEAR(result[2], 3.0, tol);
}

TEST(RigidDeriv3d, VelocityAtRotatedPointPureRotation)
{
    RigidDeriv<3, double> d(Vec<3, double>(0, 0, 0), Vec<3, double>(0, 0, 1));
    Vec<3, double> p(1, 0, 0);
    // Member: vCenter - cross(p, omega) = 0 - cross((1,0,0),(0,0,1))
    // cross((1,0,0),(0,0,1)) = (0*1-0*0, 0*0-1*1, 1*0-0*0) = (0,-1,0)
    // Result = (0,0,0) - (0,-1,0) = (0,1,0)
    auto result = d.velocityAtRotatedPoint(p);
    EXPECT_NEAR(result[0], 0.0, tol);
    EXPECT_NEAR(result[1], 1.0, tol);
    EXPECT_NEAR(result[2], 0.0, tol);
}

TEST(RigidDeriv3d, VelocityAtRotatedPointGeneral)
{
    RigidDeriv<3, double> d(Vec<3, double>(1, 0, 0), Vec<3, double>(0, 0, 2));
    Vec<3, double> p(0, 3, 0);
    // Member: (1,0,0) - cross((0,3,0),(0,0,2))
    // cross((0,3,0),(0,0,2)) = (3*2-0*0, 0*0-0*2, 0*0-3*0) = (6,0,0)
    // Result = (1,0,0) - (6,0,0) = (-5,0,0)
    auto result = d.velocityAtRotatedPoint(p);
    EXPECT_NEAR(result[0], -5.0, tol);
    EXPECT_NEAR(result[1], 0.0, tol);
    EXPECT_NEAR(result[2], 0.0, tol);
}

TEST(RigidDeriv3d, FreeVelocityAtRotatedPoint)
{
    RigidDeriv<3, double> d(Vec<3, double>(1, 0, 0), Vec<3, double>(0, 0, 2));
    Vec<3, double> p(0, 3, 0);
    // Free function: getLinear + cross(getAngular, p)
    // = (1,0,0) + cross((0,0,2),(0,3,0)) = (1,0,0) + (-6,0,0) = (-5,0,0)
    auto result = velocityAtRotatedPoint(d, p);
    EXPECT_NEAR(result[0], -5.0, tol);
    EXPECT_NEAR(result[1], 0.0, tol);
    EXPECT_NEAR(result[2], 0.0, tol);
}

// ====================================================================
// RigidDeriv<2,double> Tests
// ====================================================================

TEST(RigidDeriv2d, DefaultConstructor)
{
    RigidDeriv<2, double> d;
    EXPECT_NEAR(d[0], 0.0, tol);
    EXPECT_NEAR(d[1], 0.0, tol);
    EXPECT_NEAR(d[2], 0.0, tol);
}

TEST(RigidDeriv2d, ParameterizedConstructor)
{
    RigidDeriv<2, double> d(Vec<2, double>(1, 2), 3.0);
    EXPECT_NEAR(d.getVCenter()[0], 1.0, tol);
    EXPECT_NEAR(d.getVCenter()[1], 2.0, tol);
    EXPECT_NEAR(d.getVOrientation(), 3.0, tol);
}

TEST(RigidDeriv2d, FromVec3)
{
    Vec<3, double> v(1, 2, 3);
    RigidDeriv<2, double> d(v);
    EXPECT_NEAR(d[0], 1.0, tol);
    EXPECT_NEAR(d[1], 2.0, tol);
    EXPECT_NEAR(d[2], 3.0, tol);
}

TEST(RigidDeriv2d, Arithmetic)
{
    RigidDeriv<2, double> a(Vec<2, double>(1, 2), 3.0);
    RigidDeriv<2, double> b(Vec<2, double>(4, 5), 6.0);

    auto sum = a + b;
    EXPECT_NEAR(sum[0], 5.0, tol);
    EXPECT_NEAR(sum[1], 7.0, tol);
    EXPECT_NEAR(sum[2], 9.0, tol);

    auto diff = b - a;
    EXPECT_NEAR(diff[0], 3.0, tol);
    EXPECT_NEAR(diff[1], 3.0, tol);
    EXPECT_NEAR(diff[2], 3.0, tol);

    auto neg = -a;
    EXPECT_NEAR(neg[0], -1.0, tol);
    EXPECT_NEAR(neg[1], -2.0, tol);
    EXPECT_NEAR(neg[2], -3.0, tol);

    auto scaled = a * 2.0;
    EXPECT_NEAR(scaled[0], 2.0, tol);
    EXPECT_NEAR(scaled[1], 4.0, tol);
    EXPECT_NEAR(scaled[2], 6.0, tol);

    auto divided = a / 2.0;
    EXPECT_NEAR(divided[0], 0.5, tol);
    EXPECT_NEAR(divided[1], 1.0, tol);
    EXPECT_NEAR(divided[2], 1.5, tol);
}

TEST(RigidDeriv2d, DotProduct)
{
    RigidDeriv<2, double> a(Vec<2, double>(1, 2), 3.0);
    RigidDeriv<2, double> b(Vec<2, double>(4, 5), 6.0);
    double dot = a * b;
    EXPECT_NEAR(dot, 1.0*4 + 2.0*5 + 3.0*6, tol);
}

TEST(RigidDeriv2d, VelocityAtRotatedPointZeroOmega)
{
    RigidDeriv<2, double> d(Vec<2, double>(1, 0), 0.0);
    Vec<2, double> p(0, 3);
    auto result = d.velocityAtRotatedPoint(p);
    EXPECT_NEAR(result[0], 1.0, tol);
    EXPECT_NEAR(result[1], 0.0, tol);
}

TEST(RigidDeriv2d, VelocityAtRotatedPointGeneral)
{
    RigidDeriv<2, double> d(Vec<2, double>(1, 0), 2.0);
    Vec<2, double> p(0, 3);
    // vCenter + Vec2(-p[1], p[0]) * vOrientation
    // = (1,0) + (-3, 0) * 2 = (1,0) + (-6,0) = (-5,0)
    auto result = d.velocityAtRotatedPoint(p);
    EXPECT_NEAR(result[0], -5.0, tol);
    EXPECT_NEAR(result[1], 0.0, tol);
}

// ====================================================================
// Float Spot-Checks
// ====================================================================

TEST(RigidDeriv3f, Arithmetic)
{
    RigidDeriv<3, float> a(Vec<3, float>(1, 2, 3), Vec<3, float>(4, 5, 6));
    RigidDeriv<3, float> b(Vec<3, float>(7, 8, 9), Vec<3, float>(10, 11, 12));
    auto sum = a + b;
    EXPECT_NEAR(sum[0], 8.0f, ftol);
    EXPECT_NEAR(sum[5], 18.0f, ftol);
    float dot = a * b;
    float expected = 1.0f*7 + 2.0f*8 + 3.0f*9 + 4.0f*10 + 5.0f*11 + 6.0f*12;
    EXPECT_NEAR(dot, expected, ftol);
}

TEST(RigidDeriv2f, Arithmetic)
{
    RigidDeriv<2, float> a(Vec<2, float>(1, 2), 3.0f);
    RigidDeriv<2, float> b(Vec<2, float>(4, 5), 6.0f);
    auto sum = a + b;
    EXPECT_NEAR(sum[0], 5.0f, ftol);
    EXPECT_NEAR(sum[2], 9.0f, ftol);
    float dot = a * b;
    EXPECT_NEAR(dot, 1.0f*4 + 2.0f*5 + 3.0f*6, ftol);
}

} // anonymous namespace
