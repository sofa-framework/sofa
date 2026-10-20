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
#include <sofa/component/solidmechanics/fem/elastic/CorotationalFEMForceField.h>
#include <sofa/component/solidmechanics/testing/ForceFieldTestCreation.h>
#include <sofa/component/topology/container/constant/MeshTopology.h>

#include <cmath>
#include <numbers>
#include <type_traits>

namespace sofa
{

using namespace sofa::component::solidmechanics::fem::elastic;

namespace
{

using sofa::defaulttype::Vec2Types;
using sofa::defaulttype::Vec3Types;

template <class DataTypes>
using Matrix = sofa::type::Mat<DataTypes::spatial_dimensions, DataTypes::spatial_dimensions, SReal>;

constexpr SReal degreeToRadian = std::numbers::pi_v<SReal> / 180;

/// Helpers for computing rotations in 2D and 3D
Matrix<Vec2Types> rotation(SReal angle)
{
    return {{std::cos(angle), -std::sin(angle)}, {std::sin(angle), std::cos(angle)}};
}

Matrix<Vec3Types> rotation(const sofa::type::Vec3& axis, SReal angle)
{
    const auto K = sofa::type::crossProductMatrix(axis.normalized());
    return Matrix<Vec3Types>::Identity() + K * std::sin(angle) + K * K * (1 - std::cos(angle));
}

}

/// Checks the force derivatives against a finite difference of the force, on a single element.
template <class ForceField>
struct CorotationalFEMForceFieldTangent_test : public sofa::ForceField_test<ForceField>
{
    using DataTypes = typename ForceField::DataTypes;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;
    using Deriv = sofa::Deriv_t<DataTypes>;

    /// Rest positions of the element.
    VecCoord restPositions;

    /// Expected force produced when the element is stretched by 50% along x
    VecDeriv expectedStretchForces;

    CorotationalFEMForceFieldTangent_test()
    {
        const auto topology = sofa::core::objectmodel::New<sofa::component::topology::container::constant::MeshTopology>();
        this->node->addObject(topology);

        // E = 1 and nu = 0: the stretch stress is 0.5 along x only.
        if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec2Types, sofa::geometry::Quad>>)
        {
            topology->addQuad(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(1.0, 0.0), Coord(1.0, 1.0), Coord(0.0, 1.0)};
            expectedStretchForces = VecDeriv{Deriv(0.25, 0.0), Deriv(-0.25, 0.0), Deriv(-0.25, 0.0), Deriv(0.25, 0.0)};
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec2Types, sofa::geometry::Triangle>>)
        {
            topology->addTriangle(0, 1, 2);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(1.0, 0.0), Coord(0.0, 1.0)};
            expectedStretchForces = VecDeriv{Deriv(0.25, 0.0), Deriv(-0.25, 0.0), Deriv(0.0, 0.0)};
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec3Types, sofa::geometry::Hexahedron>>)
        {
            topology->addHexa(0, 1, 2, 3, 4, 5, 6, 7);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(1.0, 0.0, 0.0), Coord(1.0, 1.0, 0.0), Coord(0.0, 1.0, 0.0),
                                     Coord(0.0, 0.0, 1.0), Coord(1.0, 0.0, 1.0), Coord(1.0, 1.0, 1.0), Coord(0.0, 1.0, 1.0)};
            expectedStretchForces = VecDeriv{Deriv(0.125, 0.0, 0.0), Deriv(-0.125, 0.0, 0.0), Deriv(-0.125, 0.0, 0.0), Deriv(0.125, 0.0, 0.0),
                                             Deriv(0.125, 0.0, 0.0), Deriv(-0.125, 0.0, 0.0), Deriv(-0.125, 0.0, 0.0), Deriv(0.125, 0.0, 0.0)};
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec3Types, sofa::geometry::Tetrahedron>>)
        {
            topology->addTetra(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(1.0, 0.0, 0.0), Coord(0.0, 1.0, 0.0), Coord(0.0, 0.0, 1.0)};
            expectedStretchForces = VecDeriv{Deriv(1.0 / 12, 0.0, 0.0), Deriv(-1.0 / 12, 0.0, 0.0), Deriv(0.0, 0.0, 0.0), Deriv(0.0, 0.0, 0.0)};
        }
        else
        {
            static_assert(sizeof(ForceField) == 0, "No element defined for this force field");
        }

        this->dof->writeOnlyRestPositions().wref() = restPositions;

        this->force->d_youngModulus.setValue({1});
        this->force->d_poissonRatio.setValue({0});
        this->force->findData("rotationMethod")->read("polar");

        this->deltaRange = {5e5, 5e6};
        this->flags &= ~sofa::ForceField_test<ForceField>::TEST_ADD_K_TO_MATRIX;
    }

    /// Rotation by 30 degrees. In 3D, the rotation is taken about the axis (1, 2, 3).
    static Matrix<DataTypes> rotation()
    {
        if constexpr (DataTypes::spatial_dimensions == 2)
        {
            return sofa::rotation(30 * degreeToRadian);
        }
        else
        {
            return sofa::rotation(sofa::type::Vec3(1.0, 2.0, 3.0), 30 * degreeToRadian);
        }
    }

    /// Stretch by 50% along x.
    static Matrix<DataTypes> stretch()
    {
        if constexpr (DataTypes::spatial_dimensions == 2)
        {
            return {{1.5, 0.0}, {0.0, 1.0}};
        }
        else
        {
            return {{1.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
        }
    }

    /// Moves the nodes by the deformation gradient F = R U and expects the forces f rotated by R.
    void runTest(const Matrix<DataTypes>& R, const Matrix<DataTypes>& U, const VecDeriv& f)
    {
        VecCoord x(restPositions.size());
        VecDeriv v(restPositions.size());
        VecDeriv expectedForces(restPositions.size());
        for (std::size_t j = 0; j < restPositions.size(); ++j)
        {
            x[j] = R * U * restPositions[j];
            expectedForces[j] = R * f[j];
        }

        this->run_test(x, v, expectedForces);
    }
};

using TestTypes = ::testing::Types<
    CorotationalFEMForceField<Vec2Types, sofa::geometry::Quad>,
    CorotationalFEMForceField<Vec2Types, sofa::geometry::Triangle>,
    CorotationalFEMForceField<Vec3Types, sofa::geometry::Hexahedron>,
    CorotationalFEMForceField<Vec3Types, sofa::geometry::Tetrahedron>>;

TYPED_TEST_SUITE(CorotationalFEMForceFieldTangent_test, TestTypes);

TYPED_TEST(CorotationalFEMForceFieldTangent_test, rotated)
{
    using Matrix = Matrix<typename TestFixture::DataTypes>;
    this->runTest(this->rotation(), Matrix::Identity(), typename TestFixture::VecDeriv(this->restPositions.size()));
}

TYPED_TEST(CorotationalFEMForceFieldTangent_test, stretched)
{
    using Matrix = Matrix<typename TestFixture::DataTypes>;
    this->runTest(Matrix::Identity(), this->stretch(), this->expectedStretchForces);
}

TYPED_TEST(CorotationalFEMForceFieldTangent_test, stretchedRotated)
{
    this->runTest(this->rotation(), this->stretch(), this->expectedStretchForces);
}

}
