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
#include <sofa/component/solidmechanics/fem/elastic/impl/rotations/PolarDecomposition.h>
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

const Matrix<Vec2Types> stretch2D {{1.5, 0.0}, {0.0, 1.0}};
const Matrix<Vec3Types> stretch3D {{1.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

const Nodes<Vec2Types, 4> square {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0),
    sofa::type::Vec2(1.0, 1.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec2Types, 3> triangle {
    sofa::type::Vec2(0.0, 0.0), sofa::type::Vec2(1.0, 0.0), sofa::type::Vec2(0.0, 1.0)};
const Nodes<Vec3Types, 8> cube {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(1.0, 0.0, 0.0),
    sofa::type::Vec3(1.0, 1.0, 0.0), sofa::type::Vec3(0.0, 1.0, 0.0),
    sofa::type::Vec3(0.0, 0.0, 1.0), sofa::type::Vec3(1.0, 0.0, 1.0),
    sofa::type::Vec3(1.0, 1.0, 1.0), sofa::type::Vec3(0.0, 1.0, 1.0)};
const Nodes<Vec3Types, 4> tetrahedron {
    sofa::type::Vec3(0.0, 0.0, 0.0), sofa::type::Vec3(1.0, 0.0, 0.0),
    sofa::type::Vec3(0.0, 1.0, 0.0), sofa::type::Vec3(0.0, 0.0, 1.0)};

const sofa::type::Vec3 axis(1.0, 2.0, 3.0);

/// Moves the nodes by the deformation gradient F = R U and checks that PolarDecomposition extracts
/// R from their positions. The tests pass either a pure rotation, F = R, or a pure stretch, F = U
/// with R = I.
template <class DataTypes, std::size_t N>
void expectRotation(const Nodes<DataTypes, N>& restPositions, const Matrix<DataTypes>& F,
                    const Matrix<DataTypes>& expected)
{
    Nodes<DataTypes, N> positions;
    for (std::size_t j = 0; j < N; ++j)
    {
        positions[j] = F * restPositions[j];
    }

    Matrix<DataTypes> R;
    PolarDecomposition<DataTypes>().computeRotation(R, Matrix<DataTypes>::Identity(), positions, restPositions);

    for (std::size_t i = 0; i < DataTypes::spatial_dimensions; ++i)
    {
        for (std::size_t j = 0; j < DataTypes::spatial_dimensions; ++j)
        {
            EXPECT_NEAR(R(i, j), expected(i, j), 1e-10) << "R(" << i << ", " << j << ")";
        }
    }
}

}

TEST(PolarDecomposition, squareRotated)
{
    expectRotation<Vec2Types>(square, rotation(30 * degreeToRadian), rotation(30 * degreeToRadian));
}

TEST(PolarDecomposition, triangleRotated)
{
    expectRotation<Vec2Types>(triangle, rotation(30 * degreeToRadian), rotation(30 * degreeToRadian));
}

TEST(PolarDecomposition, squareStretched)
{
    expectRotation<Vec2Types>(square, stretch2D, Matrix<Vec2Types>::Identity());
}

TEST(PolarDecomposition, triangleStretched)
{
    expectRotation<Vec2Types>(triangle, stretch2D, Matrix<Vec2Types>::Identity());
}

TEST(PolarDecomposition, cubeRotated)
{
    expectRotation<Vec3Types>(cube, rotation(axis, 30 * degreeToRadian), rotation(axis, 30 * degreeToRadian));
}

TEST(PolarDecomposition, tetrahedronRotated)
{
    expectRotation<Vec3Types>(tetrahedron, rotation(axis, 30 * degreeToRadian), rotation(axis, 30 * degreeToRadian));
}

TEST(PolarDecomposition, cubeStretched)
{
    expectRotation<Vec3Types>(cube, stretch3D, Matrix<Vec3Types>::Identity());
}

TEST(PolarDecomposition, tetrahedronStretched)
{
    expectRotation<Vec3Types>(tetrahedron, stretch3D, Matrix<Vec3Types>::Identity());
}

}
