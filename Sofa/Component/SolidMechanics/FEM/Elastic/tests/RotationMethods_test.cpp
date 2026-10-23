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
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/HexahedronRotation.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/PolarDecomposition.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/StablePolarDecomposition.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/TriangleRotation.h>
#include <gtest/gtest.h>
#include <sofa/fem/FiniteElement[all].h>
#include <sofa/defaulttype/VecTypes.h>

#include <cmath>
#include <numbers>

namespace sofa
{

using namespace sofa::component::solidmechanics::fem::elastic;

namespace
{

using sofa::defaulttype::Vec2Types;
using sofa::defaulttype::Vec3Types;

template <class DataTypes>
using Matrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, sofa::Real_t<DataTypes>>;

template <class DataTypes, std::size_t N>
using Nodes = std::array<sofa::Coord_t<DataTypes>, N>;

constexpr SReal degreeToRadian = std::numbers::pi_v<SReal> / 180;

Matrix<Vec2Types> rotation(SReal angle)
{
    return {{std::cos(angle), -std::sin(angle)}, {std::sin(angle), std::cos(angle)}};
}

/// Rotation by an angle about an axis.
Matrix<Vec3Types> rotation(const sofa::type::Vec3& axis, SReal angle)
{
    const auto K = sofa::type::crossProductMatrix(axis.normalized());
    return Matrix<Vec3Types>::Identity() + K * std::sin(angle) + K * K * (1 - std::cos(angle));
}

// Shapes

/// Maps the plane z = 0 to a plane that contains no coordinate axis.
const Matrix<Vec3Types> tilt = rotation(sofa::type::Vec3(1.0, -1.0, 2.0), 40 * degreeToRadian);

/// The 2D element placed in the plane z = 0.
template <std::size_t N>
Nodes<Vec3Types, N> embedElement(const Nodes<Vec2Types, N>& nodes)
{
    Nodes<Vec3Types, N> embeddedNodes;
    for (std::size_t j = 0; j < N; ++j)
    {
        embeddedNodes[j] = sofa::type::Vec3(nodes[j][0], nodes[j][1], 0.0);
    }
    return embeddedNodes;
}

/// The 2D element placed in a tilted plane.
template <std::size_t N>
Nodes<Vec3Types, N> tiltElement(const Nodes<Vec2Types, N>& nodes)
{
    Nodes<Vec3Types, N> tiltedNodes = embedElement(nodes);
    for (auto& node : tiltedNodes)
    {
        node = tilt * node;
    }
    return tiltedNodes;
}

// Regular shapes: M = sum_j q_j q_j^T is proportional to the identity.

const Nodes<Vec2Types, 4> regularQuad2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0),
    sofa::type::Vec2(1.0, 1.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec3Types, 4> regularQuadFlat = embedElement(regularQuad2D);
const Nodes<Vec3Types, 4> regularQuadTilted = tiltElement(regularQuad2D);
const Nodes<Vec2Types, 3> regularTriangle2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0), sofa::type::Vec2(0.5, std::sqrt(3.0) / 2)};
const Nodes<Vec3Types, 3> regularTriangleFlat = embedElement(regularTriangle2D);
const Nodes<Vec3Types, 3> regularTriangleTilted = tiltElement(regularTriangle2D);
const Nodes<Vec3Types, 4> regularTetrahedron {
    sofa::type::Vec3(1.0, 1.0, 1.0), sofa::type::Vec3(1.0, -1.0, -1.0),
    sofa::type::Vec3(-1.0, 1.0, -1.0), sofa::type::Vec3(-1.0, -1.0, 1.0)};
const Nodes<Vec3Types, 8> regularHexahedron {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(1.0, 0.0, 0.0),
    sofa::type::Vec3(1.0, 1.0, 0.0), sofa::type::Vec3(0.0, 1.0, 0.0),
    sofa::type::Vec3(0.0, 0.0, 1.0), sofa::type::Vec3(1.0, 0.0, 1.0),
    sofa::type::Vec3(1.0, 1.0, 1.0), sofa::type::Vec3(0.0, 1.0, 1.0)};

// Irregular shapes: M is not proportional to the identity.

const Nodes<Vec2Types, 4> irregularQuad2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(3.0, 0.0),
    sofa::type::Vec2(3.0, 1.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec3Types, 4> irregularQuadFlat = embedElement(irregularQuad2D);
const Nodes<Vec3Types, 4> irregularQuadTilted = tiltElement(irregularQuad2D);
const Nodes<Vec2Types, 3> irregularTriangle2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec3Types, 3> irregularTriangleFlat = embedElement(irregularTriangle2D);
const Nodes<Vec3Types, 3> irregularTriangleTilted = tiltElement(irregularTriangle2D);
const Nodes<Vec3Types, 4> irregularTetrahedron {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(1.0, 0.0, 0.0),
    sofa::type::Vec3(0.0, 1.0, 0.0), sofa::type::Vec3(0.0, 0.0, 1.0)};
const Nodes<Vec3Types, 8> irregularHexahedron {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(3.0, 0.0, 0.0),
    sofa::type::Vec3(3.0, 2.0, 0.0), sofa::type::Vec3(0.0, 2.0, 0.0),
    sofa::type::Vec3(0.0, 0.0, 1.0), sofa::type::Vec3(3.0, 0.0, 1.0),
    sofa::type::Vec3(3.0, 2.0, 1.0), sofa::type::Vec3(0.0, 2.0, 1.0)};

// Rigid motion

const Matrix<Vec2Types> rotation2D = rotation(30 * degreeToRadian);
const Matrix<Vec3Types> rotation3D = rotation(sofa::type::Vec3(1.0, 2.0, 3.0), 30 * degreeToRadian);

// Stretch patch: stretches along a direction that follows no element edge

const Matrix<Vec2Types> stretch2D {{1.5, 0.0}, {0.0, 1.0}};
const Matrix<Vec3Types> stretch3D {{1.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

const Matrix<Vec2Types> offAxisStretch2D = rotation2D * stretch2D * rotation2D.transposed();
const Matrix<Vec3Types> offAxisStretch3D = rotation3D * stretch3D * rotation3D.transposed();

/// offAxisStretch2D within the plane z = 0.
const Matrix<Vec3Types> flatStretch3D = []
{
    Matrix<Vec3Types> stretch = Matrix<Vec3Types>::Identity();
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            stretch(i, j) = offAxisStretch2D(i, j);
        }
    }
    return stretch;
}();

/// offAxisStretch2D within the tilted plane.
const Matrix<Vec3Types> tiltedStretch3D = tilt * flatStretch3D * tilt.transposed();

// Renumbering: the stretches followed by a rotation, and renumberings that keep the element valid

const Matrix<Vec2Types> rotatedStretch2D = rotation2D * offAxisStretch2D;
const Matrix<Vec3Types> rotatedStretch3D = rotation3D * offAxisStretch3D;
const Matrix<Vec3Types> rotatedFlatStretch3D = rotation3D * flatStretch3D;
const Matrix<Vec3Types> rotatedTiltedStretch3D = rotation3D * tiltedStretch3D;

const std::array<std::size_t, 3> triangleNumbering {1, 2, 0};
const std::array<std::size_t, 4> quadNumbering {1, 2, 3, 0};
const std::array<std::size_t, 4> tetrahedronNumbering {1, 2, 0, 3};
const std::array<std::size_t, 8> hexahedronNumbering {1, 2, 3, 0, 5, 6, 7, 4};

/// Whether R equals the expected rotation, entry by entry, within 1e-10.
template <class RotationMatrix>
::testing::AssertionResult sameRotation(const RotationMatrix& R, const RotationMatrix& expected)
{
    for (std::size_t i = 0; i < RotationMatrix::nbLines; ++i)
    {
        for (std::size_t j = 0; j < RotationMatrix::nbCols; ++j)
        {
            if (std::abs(R(i, j) - expected(i, j)) > 1e-10)
            {
                return ::testing::AssertionFailure()
                    << "R(" << i << ", " << j << ") = " << R(i, j) << ", expected " << expected(i, j);
            }
        }
    }
    return ::testing::AssertionSuccess() << "R equals the expected rotation";
}

/// Whether R is orthogonal with determinant +1, within 1e-10.
template <class RotationMatrix>
::testing::AssertionResult isProperRotation(const RotationMatrix& R)
{
    const auto RtR = R.multTranspose(R);
    for (std::size_t i = 0; i < RotationMatrix::nbLines; ++i)
    {
        for (std::size_t j = 0; j < RotationMatrix::nbCols; ++j)
        {
            if (std::abs(RtR(i, j) - (i == j ? 1 : 0)) > 1e-10)
            {
                return ::testing::AssertionFailure()
                    << "R is not orthogonal: (R^T R)(" << i << ", " << j << ") = " << RtR(i, j);
            }
        }
    }
    if (std::abs(sofa::type::determinant(R) - 1) > 1e-10)
    {
        return ::testing::AssertionFailure() << "R is not a rotation: det R = " << sofa::type::determinant(R);
    }
    return ::testing::AssertionSuccess();
}

/// Rotation from the rest to the current positions, expected to be a proper rotation. Some methods
/// need the initial rotation, which is computed first from the rest positions.
template <class Method, class Coord, std::size_t N>
typename Method::RotationMatrix extractRotation(const std::array<Coord, N>& positions,
                                                const std::array<Coord, N>& restPositions)
{
    using RotationMatrix = typename Method::RotationMatrix;

    Method method;
    RotationMatrix initialRotation;
    method.computeRotation(initialRotation, RotationMatrix::Identity(), restPositions, restPositions);
    initialRotation.transpose();

    RotationMatrix R;
    method.computeRotation(R, initialRotation, positions, restPositions);
    EXPECT_TRUE(isProperRotation(R));
    return R;
}

/// A rigid motion x = R0 X + c gives R = R0.
template <class Method, class Coord, std::size_t N>
::testing::AssertionResult meetsRigidMotion(const std::array<Coord, N>& restPositions,
                                            const typename Method::RotationMatrix& R0)
{
    Coord c;
    for (std::size_t i = 0; i < Coord::spatial_dimensions; ++i)
    {
        c[i] = 0.5 * (i + 1);
    }

    std::array<Coord, N> positions;
    for (std::size_t j = 0; j < N; ++j)
    {
        positions[j] = R0 * restPositions[j] + c;
    }

    return sameRotation(extractRotation<Method>(positions, restPositions), R0);
}

/// A uniform stretch x = U X gives R = I.
template <class Method, class Coord, std::size_t N>
::testing::AssertionResult meetsStretchPatch(const std::array<Coord, N>& restPositions,
                                             const typename Method::RotationMatrix& U)
{
    std::array<Coord, N> positions;
    for (std::size_t j = 0; j < N; ++j)
    {
        positions[j] = U * restPositions[j];
    }

    return sameRotation(extractRotation<Method>(positions, restPositions), Method::RotationMatrix::Identity());
}

/// Renumbering the nodes of a deformed element x = F X leaves R unchanged.
template <class Method, class Coord, std::size_t N>
::testing::AssertionResult meetsRenumbering(const std::array<Coord, N>& restPositions,
                                            const typename Method::RotationMatrix& F,
                                            const std::array<std::size_t, N>& numbering)
{
    std::array<Coord, N> positions;
    for (std::size_t j = 0; j < N; ++j)
    {
        positions[j] = F * restPositions[j];
    }

    std::array<Coord, N> renumberedPositions;
    std::array<Coord, N> renumberedRestPositions;
    for (std::size_t j = 0; j < N; ++j)
    {
        renumberedPositions[j] = positions[numbering[j]];
        renumberedRestPositions[j] = restPositions[numbering[j]];
    }

    return sameRotation(extractRotation<Method>(renumberedPositions, renumberedRestPositions),
                        extractRotation<Method>(positions, restPositions));
}

}

TEST(RotationMethods, regularQuad2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types, sofa::geometry::Quad>>(regularQuad2D, rotation2D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(regularQuad2D, rotation2D));
}

TEST(RotationMethods, regularQuadFlatRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(regularQuadFlat, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularQuadFlat, rotation3D));
}

TEST(RotationMethods, regularQuadTiltedRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(regularQuadTilted, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularQuadTilted, rotation3D));
}

TEST(RotationMethods, regularTriangle2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types, sofa::geometry::Triangle>>(regularTriangle2D, rotation2D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(regularTriangle2D, rotation2D));
}

TEST(RotationMethods, regularTriangleFlatRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(regularTriangleFlat, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularTriangleFlat, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(regularTriangleFlat, rotation3D));
}

TEST(RotationMethods, regularTriangleTiltedRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(regularTriangleTilted, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularTriangleTilted, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(regularTriangleTilted, rotation3D));
}

TEST(RotationMethods, regularTetrahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Tetrahedron>>(regularTetrahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularTetrahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(regularTetrahedron, rotation3D));
}

TEST(RotationMethods, regularHexahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Hexahedron>>(regularHexahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularHexahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<HexahedronRotation<Vec3Types>>(regularHexahedron, rotation3D));
}

TEST(RotationMethods, irregularQuad2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types, sofa::geometry::Quad>>(irregularQuad2D, rotation2D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(irregularQuad2D, rotation2D));
}

TEST(RotationMethods, irregularQuadFlatRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(irregularQuadFlat, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularQuadFlat, rotation3D));
}

TEST(RotationMethods, irregularQuadTiltedRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(irregularQuadTilted, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularQuadTilted, rotation3D));
}

TEST(RotationMethods, irregularTriangle2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types, sofa::geometry::Triangle>>(irregularTriangle2D, rotation2D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(irregularTriangle2D, rotation2D));
}

TEST(RotationMethods, irregularTriangleFlatRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(irregularTriangleFlat, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularTriangleFlat, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(irregularTriangleFlat, rotation3D));
}

TEST(RotationMethods, irregularTriangleTiltedRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(irregularTriangleTilted, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularTriangleTilted, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(irregularTriangleTilted, rotation3D));
}

TEST(RotationMethods, irregularTetrahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Tetrahedron>>(irregularTetrahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularTetrahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(irregularTetrahedron, rotation3D));
}

TEST(RotationMethods, irregularHexahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types, sofa::geometry::Hexahedron>>(irregularHexahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularHexahedron, rotation3D));
    EXPECT_TRUE(meetsRigidMotion<HexahedronRotation<Vec3Types>>(irregularHexahedron, rotation3D));
}

TEST(RotationMethods, regularQuad2DStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec2Types, sofa::geometry::Quad>>(regularQuad2D, offAxisStretch2D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(regularQuad2D, offAxisStretch2D));
}

TEST(RotationMethods, regularQuadFlatStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(regularQuadFlat, flatStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularQuadFlat, flatStretch3D));
}

TEST(RotationMethods, regularQuadTiltedStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(regularQuadTilted, tiltedStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularQuadTilted, tiltedStretch3D));
}

TEST(RotationMethods, regularTriangle2DStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec2Types, sofa::geometry::Triangle>>(regularTriangle2D, offAxisStretch2D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(regularTriangle2D, offAxisStretch2D));
}

TEST(RotationMethods, regularTriangleFlatStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(regularTriangleFlat, flatStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularTriangleFlat, flatStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(regularTriangleFlat, flatStretch3D));
}

TEST(RotationMethods, regularTriangleTiltedStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(regularTriangleTilted, tiltedStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularTriangleTilted, tiltedStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(regularTriangleTilted, tiltedStretch3D));
}

TEST(RotationMethods, regularTetrahedronStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Tetrahedron>>(regularTetrahedron, offAxisStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularTetrahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(regularTetrahedron, offAxisStretch3D));
}

TEST(RotationMethods, regularHexahedronStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Hexahedron>>(regularHexahedron, offAxisStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularHexahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<HexahedronRotation<Vec3Types>>(regularHexahedron, offAxisStretch3D));
}

TEST(RotationMethods, irregularQuad2DStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec2Types, sofa::geometry::Quad>>(irregularQuad2D, offAxisStretch2D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(irregularQuad2D, offAxisStretch2D));
}

TEST(RotationMethods, irregularQuadFlatStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(irregularQuadFlat, flatStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularQuadFlat, flatStretch3D));
}

TEST(RotationMethods, irregularQuadTiltedStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(irregularQuadTilted, tiltedStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularQuadTilted, tiltedStretch3D));
}

TEST(RotationMethods, irregularTriangle2DStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec2Types, sofa::geometry::Triangle>>(irregularTriangle2D, offAxisStretch2D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(irregularTriangle2D, offAxisStretch2D));
}

TEST(RotationMethods, irregularTriangleFlatStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(irregularTriangleFlat, flatStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularTriangleFlat, flatStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(irregularTriangleFlat, flatStretch3D));
}

TEST(RotationMethods, irregularTriangleTiltedStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(irregularTriangleTilted, tiltedStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularTriangleTilted, tiltedStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(irregularTriangleTilted, tiltedStretch3D));
}

TEST(RotationMethods, irregularTetrahedronStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Tetrahedron>>(irregularTetrahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularTetrahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(irregularTetrahedron, offAxisStretch3D));
}

TEST(RotationMethods, irregularHexahedronStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types, sofa::geometry::Hexahedron>>(irregularHexahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularHexahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<HexahedronRotation<Vec3Types>>(irregularHexahedron, offAxisStretch3D));
}

TEST(RotationMethods, regularQuad2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types, sofa::geometry::Quad>>(regularQuad2D, rotatedStretch2D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(regularQuad2D, rotatedStretch2D, quadNumbering));
}

TEST(RotationMethods, regularQuadFlatRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(regularQuadFlat, rotatedFlatStretch3D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularQuadFlat, rotatedFlatStretch3D, quadNumbering));
}

TEST(RotationMethods, regularQuadTiltedRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(regularQuadTilted, rotatedTiltedStretch3D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularQuadTilted, rotatedTiltedStretch3D, quadNumbering));
}

TEST(RotationMethods, regularTriangle2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types, sofa::geometry::Triangle>>(regularTriangle2D, rotatedStretch2D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(regularTriangle2D, rotatedStretch2D, triangleNumbering));
}

TEST(RotationMethods, regularTriangleFlatRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(regularTriangleFlat, rotatedFlatStretch3D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularTriangleFlat, rotatedFlatStretch3D, triangleNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(regularTriangleFlat, rotatedFlatStretch3D, triangleNumbering));
}

TEST(RotationMethods, regularTriangleTiltedRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(regularTriangleTilted, rotatedTiltedStretch3D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularTriangleTilted, rotatedTiltedStretch3D, triangleNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(regularTriangleTilted, rotatedTiltedStretch3D, triangleNumbering));
}

TEST(RotationMethods, regularTetrahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Tetrahedron>>(regularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(regularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
}

TEST(RotationMethods, regularHexahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Hexahedron>>(regularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<HexahedronRotation<Vec3Types>>(regularHexahedron, rotatedStretch3D, hexahedronNumbering));
}

TEST(RotationMethods, irregularQuad2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types, sofa::geometry::Quad>>(irregularQuad2D, rotatedStretch2D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(irregularQuad2D, rotatedStretch2D, quadNumbering));
}

TEST(RotationMethods, irregularQuadFlatRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(irregularQuadFlat, rotatedFlatStretch3D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularQuadFlat, rotatedFlatStretch3D, quadNumbering));
}

TEST(RotationMethods, irregularQuadTiltedRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Quad>>(irregularQuadTilted, rotatedTiltedStretch3D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularQuadTilted, rotatedTiltedStretch3D, quadNumbering));
}

TEST(RotationMethods, irregularTriangle2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types, sofa::geometry::Triangle>>(irregularTriangle2D, rotatedStretch2D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(irregularTriangle2D, rotatedStretch2D, triangleNumbering));
}

TEST(RotationMethods, irregularTriangleFlatRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(irregularTriangleFlat, rotatedFlatStretch3D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularTriangleFlat, rotatedFlatStretch3D, triangleNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(irregularTriangleFlat, rotatedFlatStretch3D, triangleNumbering));
}

TEST(RotationMethods, irregularTriangleTiltedRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Triangle>>(irregularTriangleTilted, rotatedTiltedStretch3D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularTriangleTilted, rotatedTiltedStretch3D, triangleNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(irregularTriangleTilted, rotatedTiltedStretch3D, triangleNumbering));
}

TEST(RotationMethods, irregularTetrahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Tetrahedron>>(irregularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(irregularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
}

TEST(RotationMethods, irregularHexahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types, sofa::geometry::Hexahedron>>(irregularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<HexahedronRotation<Vec3Types>>(irregularHexahedron, rotatedStretch3D, hexahedronNumbering));
}

}
