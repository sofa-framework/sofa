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
#include <cmath>

namespace
{

using namespace sofa::defaulttype;
using sofa::type::Vec;

using Mass3d = RigidMass<3, double>;
using Mass2d = RigidMass<2, double>;

constexpr double tol = 1e-10;
constexpr float ftol = 1e-5f;

// ====================================================================
// RigidMass<3,double> Tests
// ====================================================================

TEST(RigidMass3d, DefaultConstruction)
{
    RigidMass<3, double> m;
    EXPECT_NEAR(m.mass, 1.0, tol);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(m.inertiaMassMatrix(i, j), (i == j) ? 1.0 : 0.0, tol);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(m.invInertiaMassMatrix(i, j), (i == j) ? 1.0 : 0.0, tol);
}

TEST(RigidMass3d, ConstructWithMass)
{
    RigidMass<3, double> m(5.0);
    EXPECT_NEAR(m.mass, 5.0, tol);
    // inertiaMassMatrix = identity * 5 = diag(5,5,5)
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(m.inertiaMassMatrix(i, i), 5.0, tol);
    // invInertiaMassMatrix = diag(1/5)
    for (int i = 0; i < 3; ++i)
        EXPECT_NEAR(m.invInertiaMassMatrix(i, i), 0.2, tol);
}

TEST(RigidMass3d, Recalc)
{
    RigidMass<3, double> m;
    m.mass = 2.0;
    m.inertiaMatrix.identity();
    m.inertiaMatrix(0, 0) = 2.0;
    m.inertiaMatrix(1, 1) = 3.0;
    m.inertiaMatrix(2, 2) = 4.0;
    m.recalc();
    // inertiaMassMatrix = diag(2,3,4) * 2 = diag(4,6,8)
    EXPECT_NEAR(m.inertiaMassMatrix(0, 0), 4.0, tol);
    EXPECT_NEAR(m.inertiaMassMatrix(1, 1), 6.0, tol);
    EXPECT_NEAR(m.inertiaMassMatrix(2, 2), 8.0, tol);
}

TEST(RigidMass3d, AssignmentOperators)
{
    RigidMass<3, double> m;
    m = 3.0;
    EXPECT_NEAR(m.mass, 3.0, tol);
    m += 2.0;
    EXPECT_NEAR(m.mass, 5.0, tol);
    m -= 1.0;
    EXPECT_NEAR(m.mass, 4.0, tol);
}

TEST(RigidMass3d, MultiplyEquals)
{
    RigidMass<3, double> m(2.0);
    // Initially: mass=2, inertiaMassMatrix = 2*I, invInertiaMassMatrix = 0.5*I
    m *= 2.0;
    EXPECT_NEAR(m.mass, 4.0, tol);
    EXPECT_NEAR(m.inertiaMassMatrix(0, 0), 4.0, tol);
    EXPECT_NEAR(m.invInertiaMassMatrix(0, 0), 0.25, tol);
}

TEST(RigidMass3d, CastToReal)
{
    RigidMass<3, double> m(7.5);
    double x = m;
    EXPECT_NEAR(x, 7.5, tol);
}

TEST(RigidMass3d, DerivTimesRigidMass)
{
    RigidMass<3, double> m(2.0);
    RigidDeriv<3, double> d(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    auto result = d * m;
    // linear scaled by mass
    EXPECT_NEAR(result.getVCenter()[0], 2.0, tol);
    EXPECT_NEAR(result.getVCenter()[1], 4.0, tol);
    EXPECT_NEAR(result.getVCenter()[2], 6.0, tol);
    // angular scaled by inertiaMassMatrix (2*I)
    EXPECT_NEAR(result.getVOrientation()[0], 8.0, tol);
    EXPECT_NEAR(result.getVOrientation()[1], 10.0, tol);
    EXPECT_NEAR(result.getVOrientation()[2], 12.0, tol);
}

TEST(RigidMass3d, RigidMassTimesDeriv)
{
    RigidMass<3, double> m(2.0);
    RigidDeriv<3, double> d(Vec<3, double>(1, 2, 3), Vec<3, double>(4, 5, 6));
    auto result = m * d;
    EXPECT_NEAR(result.getVCenter()[0], 2.0, tol);
    EXPECT_NEAR(result.getVCenter()[1], 4.0, tol);
    EXPECT_NEAR(result.getVCenter()[2], 6.0, tol);
    EXPECT_NEAR(result.getVOrientation()[0], 8.0, tol);
    EXPECT_NEAR(result.getVOrientation()[1], 10.0, tol);
    EXPECT_NEAR(result.getVOrientation()[2], 12.0, tol);
}

TEST(RigidMass3d, DerivDivideRigidMass)
{
    RigidMass<3, double> m(2.0);
    RigidDeriv<3, double> d(Vec<3, double>(4, 6, 8), Vec<3, double>(10, 12, 14));
    auto result = d / m;
    // linear divided by mass
    EXPECT_NEAR(result.getVCenter()[0], 2.0, tol);
    EXPECT_NEAR(result.getVCenter()[1], 3.0, tol);
    EXPECT_NEAR(result.getVCenter()[2], 4.0, tol);
    // angular multiplied by invInertiaMassMatrix (0.5*I)
    EXPECT_NEAR(result.getVOrientation()[0], 5.0, tol);
    EXPECT_NEAR(result.getVOrientation()[1], 6.0, tol);
    EXPECT_NEAR(result.getVOrientation()[2], 7.0, tol);
}

TEST(RigidMass3d, NonTrivialInertiaProduct)
{
    RigidMass<3, double> m;
    m.mass = 1.0;
    m.inertiaMatrix.identity();
    m.inertiaMatrix(0, 0) = 2.0;
    m.inertiaMatrix(1, 1) = 3.0;
    m.inertiaMatrix(2, 2) = 4.0;
    m.recalc();

    RigidDeriv<3, double> d(Vec<3, double>(1, 1, 1), Vec<3, double>(1, 1, 1));
    auto result = d * m;
    // linear: (1,1,1) * 1 = (1,1,1)
    EXPECT_NEAR(result.getVCenter()[0], 1.0, tol);
    EXPECT_NEAR(result.getVCenter()[1], 1.0, tol);
    EXPECT_NEAR(result.getVCenter()[2], 1.0, tol);
    // angular: diag(2,3,4) * (1,1,1) = (2,3,4)
    EXPECT_NEAR(result.getVOrientation()[0], 2.0, tol);
    EXPECT_NEAR(result.getVOrientation()[1], 3.0, tol);
    EXPECT_NEAR(result.getVOrientation()[2], 4.0, tol);
}

// ====================================================================
// RigidMass<2,double> Tests
// ====================================================================

TEST(RigidMass2d, DefaultConstruction)
{
    RigidMass<2, double> m;
    EXPECT_NEAR(m.mass, 1.0, tol);
    EXPECT_NEAR(m.inertiaMatrix, 1.0, tol);
    EXPECT_NEAR(m.inertiaMassMatrix, 1.0, tol);
    EXPECT_NEAR(m.invInertiaMassMatrix, 1.0, tol);
}

TEST(RigidMass2d, CircleConstructor)
{
    double mass = 5.0;
    double radius = 2.0;
    RigidMass<2, double> m(mass, radius);
    EXPECT_NEAR(m.mass, 5.0, tol);
    EXPECT_NEAR(m.inertiaMatrix, 2.0, tol);  // r^2/2 = 4/2 = 2
    EXPECT_NEAR(m.inertiaMassMatrix, 10.0, tol);  // 2 * 5 = 10
    EXPECT_NEAR(m.invInertiaMassMatrix, 0.1, tol);  // 1/10
}

TEST(RigidMass2d, RectangleConstructor)
{
    double mass = 6.0;
    double xw = 3.0, yw = 4.0;
    RigidMass<2, double> m(mass, xw, yw);
    EXPECT_NEAR(m.mass, 6.0, tol);
    EXPECT_NEAR(m.volume, 12.0, tol);  // area = 3 * 4 = 12
    // inertia = (xw^2 + yw^2) / 12 = (9+16)/12 = 25/12
    EXPECT_NEAR(m.inertiaMatrix, 25.0 / 12.0, tol);
    EXPECT_NEAR(m.inertiaMassMatrix, 25.0 / 12.0 * 6.0, tol);
}

TEST(RigidMass2d, ZeroMassThrows)
{
    EXPECT_THROW(Mass2d(0.0), std::runtime_error);
}

TEST(RigidMass2d, AssignmentOperators)
{
    RigidMass<2, double> m;
    m = 3.0;
    EXPECT_NEAR(m.mass, 3.0, tol);
    m += 2.0;
    EXPECT_NEAR(m.mass, 5.0, tol);
    m -= 1.0;
    EXPECT_NEAR(m.mass, 4.0, tol);
}

TEST(RigidMass2d, MultiplyEquals)
{
    RigidMass<2, double> m(2.0);
    // Initially: mass=2, inertiaMassMatrix=2, invInertiaMassMatrix=0.5
    m *= 3.0;
    EXPECT_NEAR(m.mass, 6.0, tol);
    EXPECT_NEAR(m.inertiaMassMatrix, 6.0, tol);
    EXPECT_NEAR(m.invInertiaMassMatrix, 1.0 / 6.0, tol);
}

TEST(RigidMass2d, DerivTimesRigidMass)
{
    RigidMass<2, double> m(3.0);
    RigidDeriv<2, double> d(Vec<2, double>(2, 4), 5.0);
    auto result = d * m;
    // linear: (2,4) * 3 = (6,12)
    EXPECT_NEAR(result.getVCenter()[0], 6.0, tol);
    EXPECT_NEAR(result.getVCenter()[1], 12.0, tol);
    // angular: inertiaMassMatrix * 5 = 3 * 5 = 15
    EXPECT_NEAR(result.getVOrientation(), 15.0, tol);
}

TEST(RigidMass2d, DerivDivideRigidMass)
{
    RigidMass<2, double> m(2.0);
    RigidDeriv<2, double> d(Vec<2, double>(4, 6), 10.0);
    auto result = d / m;
    // linear: (4,6) / 2 = (2,3)
    EXPECT_NEAR(result.getVCenter()[0], 2.0, tol);
    EXPECT_NEAR(result.getVCenter()[1], 3.0, tol);
    // angular: invInertiaMassMatrix * 10 = 0.5 * 10 = 5
    EXPECT_NEAR(result.getVOrientation(), 5.0, tol);
}

// ====================================================================
// Stream Roundtrip Tests
// ====================================================================

TEST(RigidMass3d, StreamRoundtrip)
{
    RigidMass<3, double> original;
    original.mass = 3.0;
    original.volume = 2.5;
    original.inertiaMatrix.identity();
    original.inertiaMatrix(0, 0) = 2.0;
    original.inertiaMatrix(1, 1) = 3.0;
    original.inertiaMatrix(2, 2) = 4.0;
    original.recalc();

    std::stringstream ss;
    ss << original;

    RigidMass<3, double> recovered;
    ss >> recovered;

    EXPECT_NEAR(recovered.mass, 3.0, tol);
    EXPECT_NEAR(recovered.volume, 2.5, tol);
    // inertiaMassMatrix should be recomputed: diag(2,3,4) * 3 = diag(6,9,12)
    EXPECT_NEAR(recovered.inertiaMassMatrix(0, 0), 6.0, tol);
    EXPECT_NEAR(recovered.inertiaMassMatrix(1, 1), 9.0, tol);
    EXPECT_NEAR(recovered.inertiaMassMatrix(2, 2), 12.0, tol);
    // invInertiaMassMatrix should be recomputed: diag(1/6, 1/9, 1/12)
    EXPECT_NEAR(recovered.invInertiaMassMatrix(0, 0), 1.0 / 6.0, tol);
    EXPECT_NEAR(recovered.invInertiaMassMatrix(1, 1), 1.0 / 9.0, tol);
    EXPECT_NEAR(recovered.invInertiaMassMatrix(2, 2), 1.0 / 12.0, tol);
}

TEST(RigidMass2d, StreamRoundtrip)
{
    RigidMass<2, double> original(5.0, 2.0); // circle: mass=5, radius=2
    // inertiaMatrix = r^2/2 = 2, inertiaMassMatrix = 10

    std::stringstream ss;
    ss << original;

    RigidMass<2, double> recovered;
    ss >> recovered;

    EXPECT_NEAR(recovered.mass, 5.0, tol);
    EXPECT_NEAR(recovered.inertiaMatrix, 2.0, tol);
    // inertiaMassMatrix should be recomputed: 2 * 5 = 10
    EXPECT_NEAR(recovered.inertiaMassMatrix, 10.0, tol);
    // invInertiaMassMatrix should be recomputed: 1/10 = 0.1
    EXPECT_NEAR(recovered.invInertiaMassMatrix, 0.1, tol);
}

// ====================================================================
// Float Spot-Checks
// ====================================================================

TEST(RigidMass3f, Product)
{
    RigidMass<3, float> m(2.0f);
    RigidDeriv<3, float> d(Vec<3, float>(1, 2, 3), Vec<3, float>(4, 5, 6));
    auto result = d * m;
    EXPECT_NEAR(result.getVCenter()[0], 2.0f, ftol);
    EXPECT_NEAR(result.getVOrientation()[0], 8.0f, ftol);
}

TEST(RigidMass2f, Product)
{
    RigidMass<2, float> m(3.0f);
    RigidDeriv<2, float> d(Vec<2, float>(2, 4), 5.0f);
    auto result = d * m;
    EXPECT_NEAR(result.getVCenter()[0], 6.0f, ftol);
    EXPECT_NEAR(result.getVOrientation(), 15.0f, ftol);
}

} // anonymous namespace
