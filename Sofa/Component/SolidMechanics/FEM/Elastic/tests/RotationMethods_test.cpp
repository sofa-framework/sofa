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
using Matrix = typename PolarDecomposition<DataTypes>::RotationMatrix;

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

const sofa::type::Vec3 axis(1.0, 2.0, 3.0);

/// Maps the plane z = 0 to a plane that contains no coordinate axis.
const Matrix<Vec3Types> tilt = rotation(sofa::type::Vec3(1.0, -1.0, 2.0), 40 * degreeToRadian);

const Matrix<Vec2Types> stretch2D {{1.5, 0.0}, {0.0, 1.0}};
const Matrix<Vec3Types> stretch3D {{1.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

// Stretches along a direction that follows no element edge.
const Matrix<Vec2Types> offAxisStretch2D =
    rotation(30 * degreeToRadian) * stretch2D * rotation(30 * degreeToRadian).transposed();
const Matrix<Vec3Types> offAxisStretch3D =
    rotation(axis, 30 * degreeToRadian) * stretch3D * rotation(axis, 30 * degreeToRadian).transposed();

/// offAxisStretch2D within the tilted plane.
const Matrix<Vec3Types> inPlaneStretch3D = []
{
    Matrix<Vec3Types> stretch = Matrix<Vec3Types>::Identity();
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            stretch(i, j) = offAxisStretch2D(i, j);
        }
    }
    return tilt * stretch * tilt.transposed();
}();

// Stretches followed by a rotation.
const Matrix<Vec2Types> rotatedStretch2D = rotation(30 * degreeToRadian) * offAxisStretch2D;
const Matrix<Vec3Types> rotatedStretch3D = rotation(axis, 30 * degreeToRadian) * offAxisStretch3D;
const Matrix<Vec3Types> rotatedInPlaneStretch3D = rotation(axis, 30 * degreeToRadian) * inPlaneStretch3D;

// Renumberings that keep the element valid.
const std::array<std::size_t, 3> triangleNumbering {1, 2, 0};
const std::array<std::size_t, 4> quadNumbering {1, 2, 3, 0};
const std::array<std::size_t, 4> tetrahedronNumbering {1, 2, 0, 3};
const std::array<std::size_t, 8> hexahedronNumbering {1, 2, 3, 0, 5, 6, 7, 4};

/// The 2D element placed in the tilted plane.
template <std::size_t N>
Nodes<Vec3Types, N> tiltElement(const Nodes<Vec2Types, N>& nodes)
{
    Nodes<Vec3Types, N> tiltedNodes;
    for (std::size_t j = 0; j < N; ++j)
    {
        tiltedNodes[j] = tilt * sofa::type::Vec3(nodes[j][0], nodes[j][1], 0.0);
    }
    return tiltedNodes;
}

// Regular shapes: M = sum_j q_j q_j^T is proportional to the identity.

const Nodes<Vec2Types, 4> regularQuad2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0),
    sofa::type::Vec2(1.0, 1.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec3Types, 4> regularQuad3D = tiltElement(regularQuad2D);
const Nodes<Vec2Types, 3> regularTriangle2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0), sofa::type::Vec2(0.5, std::sqrt(3.0) / 2)};
const Nodes<Vec3Types, 3> regularTriangle3D = tiltElement(regularTriangle2D);
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
const Nodes<Vec3Types, 4> irregularQuad3D = tiltElement(irregularQuad2D);
const Nodes<Vec2Types, 3> irregularTriangle2D {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec3Types, 3> irregularTriangle3D = tiltElement(irregularTriangle2D);
const Nodes<Vec3Types, 4> irregularTetrahedron {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(1.0, 0.0, 0.0),
    sofa::type::Vec3(0.0, 1.0, 0.0), sofa::type::Vec3(0.0, 0.0, 1.0)};
const Nodes<Vec3Types, 8> irregularHexahedron {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(3.0, 0.0, 0.0),
    sofa::type::Vec3(3.0, 2.0, 0.0), sofa::type::Vec3(0.0, 2.0, 0.0),
    sofa::type::Vec3(0.0, 0.0, 1.0), sofa::type::Vec3(3.0, 0.0, 1.0),
    sofa::type::Vec3(3.0, 2.0, 1.0), sofa::type::Vec3(0.0, 2.0, 1.0)};

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

/// Rotation from the rest to the current positions. Some methods need the initial rotation, which
/// is computed first from the rest positions.
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
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types>>(regularQuad2D, rotation(30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(regularQuad2D, rotation(30 * degreeToRadian)));
}

TEST(RotationMethods, regularQuad3DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(regularQuad3D, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularQuad3D, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, regularTriangle2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types>>(regularTriangle2D, rotation(30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(regularTriangle2D, rotation(30 * degreeToRadian)));
}

TEST(RotationMethods, regularTriangle3DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(regularTriangle3D, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularTriangle3D, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(regularTriangle3D, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, regularTetrahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(regularTetrahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularTetrahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(regularTetrahedron, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, regularHexahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(regularHexahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(regularHexahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<HexahedronRotation<Vec3Types>>(regularHexahedron, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, irregularQuad2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types>>(irregularQuad2D, rotation(30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(irregularQuad2D, rotation(30 * degreeToRadian)));
}

TEST(RotationMethods, irregularQuad3DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(irregularQuad3D, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularQuad3D, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, irregularTriangle2DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec2Types>>(irregularTriangle2D, rotation(30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec2Types>>(irregularTriangle2D, rotation(30 * degreeToRadian)));
}

TEST(RotationMethods, irregularTriangle3DRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(irregularTriangle3D, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularTriangle3D, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(irregularTriangle3D, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, irregularTetrahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(irregularTetrahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularTetrahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<TriangleRotation<Vec3Types>>(irregularTetrahedron, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, irregularHexahedronRigidMotion)
{
    EXPECT_TRUE(meetsRigidMotion<PolarDecomposition<Vec3Types>>(irregularHexahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<StablePolarDecomposition<Vec3Types>>(irregularHexahedron, rotation(axis, 30 * degreeToRadian)));
    EXPECT_TRUE(meetsRigidMotion<HexahedronRotation<Vec3Types>>(irregularHexahedron, rotation(axis, 30 * degreeToRadian)));
}

TEST(RotationMethods, regularQuad2DStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec2Types>>(regularQuad2D, offAxisStretch2D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(regularQuad2D, offAxisStretch2D));
}

TEST(RotationMethods, regularQuad3DStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(regularQuad3D, inPlaneStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularQuad3D, inPlaneStretch3D));
}

TEST(RotationMethods, regularTriangle2DStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec2Types>>(regularTriangle2D, offAxisStretch2D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(regularTriangle2D, offAxisStretch2D));
}

TEST(RotationMethods, regularTriangle3DStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(regularTriangle3D, inPlaneStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularTriangle3D, inPlaneStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(regularTriangle3D, inPlaneStretch3D));
}

TEST(RotationMethods, regularTetrahedronStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(regularTetrahedron, offAxisStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularTetrahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(regularTetrahedron, offAxisStretch3D));
}

TEST(RotationMethods, regularHexahedronStretchPatch)
{
    EXPECT_TRUE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(regularHexahedron, offAxisStretch3D));
    EXPECT_TRUE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(regularHexahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<HexahedronRotation<Vec3Types>>(regularHexahedron, offAxisStretch3D));
}

TEST(RotationMethods, irregularQuad2DStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec2Types>>(irregularQuad2D, offAxisStretch2D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(irregularQuad2D, offAxisStretch2D));
}

TEST(RotationMethods, irregularQuad3DStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(irregularQuad3D, inPlaneStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularQuad3D, inPlaneStretch3D));
}

TEST(RotationMethods, irregularTriangle2DStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec2Types>>(irregularTriangle2D, offAxisStretch2D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec2Types>>(irregularTriangle2D, offAxisStretch2D));
}

TEST(RotationMethods, irregularTriangle3DStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(irregularTriangle3D, inPlaneStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularTriangle3D, inPlaneStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(irregularTriangle3D, inPlaneStretch3D));
}

TEST(RotationMethods, irregularTetrahedronStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(irregularTetrahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularTetrahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<TriangleRotation<Vec3Types>>(irregularTetrahedron, offAxisStretch3D));
}

TEST(RotationMethods, irregularHexahedronStretchPatch)
{
    EXPECT_FALSE(meetsStretchPatch<PolarDecomposition<Vec3Types>>(irregularHexahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<StablePolarDecomposition<Vec3Types>>(irregularHexahedron, offAxisStretch3D));
    EXPECT_FALSE(meetsStretchPatch<HexahedronRotation<Vec3Types>>(irregularHexahedron, offAxisStretch3D));
}

TEST(RotationMethods, regularQuad2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types>>(regularQuad2D, rotatedStretch2D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(regularQuad2D, rotatedStretch2D, quadNumbering));
}

TEST(RotationMethods, regularQuad3DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(regularQuad3D, rotatedInPlaneStretch3D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularQuad3D, rotatedInPlaneStretch3D, quadNumbering));
}

TEST(RotationMethods, regularTriangle2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types>>(regularTriangle2D, rotatedStretch2D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(regularTriangle2D, rotatedStretch2D, triangleNumbering));
}

TEST(RotationMethods, regularTriangle3DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(regularTriangle3D, rotatedInPlaneStretch3D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularTriangle3D, rotatedInPlaneStretch3D, triangleNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(regularTriangle3D, rotatedInPlaneStretch3D, triangleNumbering));
}

TEST(RotationMethods, regularTetrahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(regularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(regularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
}

TEST(RotationMethods, regularHexahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(regularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(regularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<HexahedronRotation<Vec3Types>>(regularHexahedron, rotatedStretch3D, hexahedronNumbering));
}

TEST(RotationMethods, irregularQuad2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types>>(irregularQuad2D, rotatedStretch2D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(irregularQuad2D, rotatedStretch2D, quadNumbering));
}

TEST(RotationMethods, irregularQuad3DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(irregularQuad3D, rotatedInPlaneStretch3D, quadNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularQuad3D, rotatedInPlaneStretch3D, quadNumbering));
}

TEST(RotationMethods, irregularTriangle2DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec2Types>>(irregularTriangle2D, rotatedStretch2D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec2Types>>(irregularTriangle2D, rotatedStretch2D, triangleNumbering));
}

TEST(RotationMethods, irregularTriangle3DRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(irregularTriangle3D, rotatedInPlaneStretch3D, triangleNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularTriangle3D, rotatedInPlaneStretch3D, triangleNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(irregularTriangle3D, rotatedInPlaneStretch3D, triangleNumbering));
}

TEST(RotationMethods, irregularTetrahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(irregularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<TriangleRotation<Vec3Types>>(irregularTetrahedron, rotatedStretch3D, tetrahedronNumbering));
}

TEST(RotationMethods, irregularHexahedronRenumbering)
{
    EXPECT_TRUE(meetsRenumbering<PolarDecomposition<Vec3Types>>(irregularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_TRUE(meetsRenumbering<StablePolarDecomposition<Vec3Types>>(irregularHexahedron, rotatedStretch3D, hexahedronNumbering));
    EXPECT_FALSE(meetsRenumbering<HexahedronRotation<Vec3Types>>(irregularHexahedron, rotatedStretch3D, hexahedronNumbering));
}

}
