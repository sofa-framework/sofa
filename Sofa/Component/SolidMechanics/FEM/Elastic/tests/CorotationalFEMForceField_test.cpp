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
#include <string>
#include <type_traits>
#include <vector>

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

/// Helpers for computing rotation matrices in 2D and 3D
Matrix<Vec2Types> rotationMatrix2D(SReal angle)
{
    return {{std::cos(angle), -std::sin(angle)}, {std::sin(angle), std::cos(angle)}};
}

Matrix<Vec3Types> rotationMatrix3D(const sofa::type::Vec3& axis, SReal angle)
{
    const auto K = sofa::type::crossProductMatrix(axis.normalized());
    return Matrix<Vec3Types>::Identity() + K * std::sin(angle) + K * K * (1 - std::cos(angle));
}

// The matrix for the applied rotation in 2D and 3D
const Matrix<Vec2Types> prescribedRotationMatrix2D = rotationMatrix2D(30 * degreeToRadian);
const Matrix<Vec3Types> prescribedRotationMatrix3D = rotationMatrix3D(sofa::type::Vec3(1.0, 2.0, 3.0), 30 * degreeToRadian);

// Stretches by 50% along the axes 
// (0.866, 0.5) in 2D and
// (0.876, 0.420, -0.239) in 3D
// This axis was chosen so that it does not align with any edge of the test elements. If placed
// along an edge, the "triangle" and "hexahedron" rotation methods can extract the exact rotation
// by coincidence.
const Matrix<Vec2Types> prescribedStretchMatrix2D =
    prescribedRotationMatrix2D * Matrix<Vec2Types>{{1.5, 0.0}, {0.0, 1.0}} * prescribedRotationMatrix2D.transposed();
const Matrix<Vec3Types> prescribedStretchMatrix3D =
    prescribedRotationMatrix3D * Matrix<Vec3Types>{{1.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}} * prescribedRotationMatrix3D.transposed();

/// prescribedStretchMatrix2D within the plane z = 0, for the triangles and quads in 3D.
const Matrix<Vec3Types> prescribedFlatStretchMatrix3D {
    {prescribedStretchMatrix2D(0, 0), prescribedStretchMatrix2D(0, 1), 0.0},
    {prescribedStretchMatrix2D(1, 0), prescribedStretchMatrix2D(1, 1), 0.0},
    {0.0, 0.0, 1.0}};

// An edge is stretched along its own line. any other direction involves rotation.
const Matrix<Vec2Types> prescribedEdgeStretchMatrix2D {{1.5, 0.0}, {0.0, 1.0}};

}

/// Checks the force derivatives against a finite difference of the force, on a single element.
template <class ForceField>
struct CorotationalFEMForceFieldTangent_test : public sofa::ForceField_test<ForceField>
{
    using DataTypes = typename ForceField::DataTypes;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;

    /// Rest positions of the element. 
    VecCoord restPositions;

    Matrix<DataTypes> rotation;
    Matrix<DataTypes> stretch;

    /// The rotation methods available for the element, except identity, which does not rotate.
    std::vector<std::string> rotationMethods {"polar", "polar_affine", "stable_polar", "stable_polar_affine"};

    CorotationalFEMForceFieldTangent_test()
    {
        const auto topology = sofa::core::objectmodel::New<sofa::component::topology::container::constant::MeshTopology>();
        this->node->addObject(topology);

        /// An irregular shape is chosen for all elements that follow. This skews the rigid fit of
        /// the polar decomposition methods. Rigid and affine fits extract different rotations and
        /// the tangent of each must follow how its own rotation changes.
        if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec2Types, sofa::geometry::Quad>>)
        {
            topology->addQuad(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(3.0, 0.0), Coord(3.0, 1.0), Coord(0.0, 1.0)};
            rotation = prescribedRotationMatrix2D;
            stretch = prescribedStretchMatrix2D;
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec2Types, sofa::geometry::Triangle>>)
        {
            topology->addTriangle(0, 1, 2);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(1.0, 0.0), Coord(0.0, 1.0)};
            rotation = prescribedRotationMatrix2D;
            stretch = prescribedStretchMatrix2D;
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec2Types, sofa::geometry::Edge>>)
        {
            topology->addEdge(0, 1);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(1.0, 0.0)};
            rotation = prescribedRotationMatrix2D;
            stretch = prescribedEdgeStretchMatrix2D;
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec3Types, sofa::geometry::Quad>>)
        {
            topology->addQuad(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(3.0, 0.0, 0.0), Coord(3.0, 1.0, 0.0), Coord(0.0, 1.0, 0.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedFlatStretchMatrix3D;
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec3Types, sofa::geometry::Triangle>>)
        {
            topology->addTriangle(0, 1, 2);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(1.0, 0.0, 0.0), Coord(0.0, 1.0, 0.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedFlatStretchMatrix3D;
            rotationMethods.push_back("triangle");
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec3Types, sofa::geometry::Hexahedron>>)
        {
            topology->addHexa(0, 1, 2, 3, 4, 5, 6, 7);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(3.0, 0.0, 0.0), Coord(3.0, 2.0, 0.0), Coord(0.0, 2.0, 0.0),
                                     Coord(0.0, 0.0, 1.0), Coord(3.0, 0.0, 1.0), Coord(3.0, 2.0, 1.0), Coord(0.0, 2.0, 1.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedStretchMatrix3D;
            rotationMethods.push_back("hexahedron");
        }
        else if constexpr (std::is_same_v<ForceField, CorotationalFEMForceField<Vec3Types, sofa::geometry::Tetrahedron>>)
        {
            topology->addTetra(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(1.0, 0.0, 0.0), Coord(0.0, 1.0, 0.0), Coord(0.0, 0.0, 1.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedStretchMatrix3D;
            rotationMethods.push_back("triangle");
        }
        else
        {
            static_assert(sizeof(ForceField) == 0, "No element defined for this force field");
        }

        this->dof->writeOnlyRestPositions().wref() = restPositions;

        this->force->d_youngModulus.setValue({1});
        this->force->d_poissonRatio.setValue({0});

        this->deltaRange = {5e5, 5e6};
        this->flags &= ~sofa::ForceField_test<ForceField>::TEST_ADD_K_TO_MATRIX;
    }

    /// The forces of the force field at the positions x.
    VecDeriv forcesAt(const VecCoord& x, const VecDeriv& v)
    {
        this->setupState(x, v);
        sofa::simulation::node::initRoot(this->node.get());
        core::MechanicalParams mparams;
        this->computeForce(&mparams);
        VecDeriv forces;
        sofa::testing::copyFromData(forces, this->dof->readForces());
        return forces;
    }

    /// Moves the nodes by the deformation gradient F = R U and checks each rotation method.
    void runTest(const Matrix<DataTypes>& R, const Matrix<DataTypes>& U)
    {
        // The element at rest, stretched by U, then rotated by R.
        VecCoord stretchedPositions(restPositions.size());
        VecCoord x(restPositions.size());
        VecDeriv v(restPositions.size());
        for (std::size_t j = 0; j < restPositions.size(); ++j)
        {
            stretchedPositions[j] = U * restPositions[j];
            x[j] = R * stretchedPositions[j];
        }

        // The expected forces: the linear elastic forces of the stretch, computed by the identity
        // method, rotated by R.
        this->force->findData("rotationMethod")->read("identity");
        VecDeriv expectedForces = forcesAt(stretchedPositions, v);
        for (auto& force : expectedForces)
        {
            force = R * force;
        }

        // This test checks the consistency of the tangent. The elements have irregular shapes, on which
        // only the polar decomposition of the affine fit extracts the rotation that produces the correct
        // forces. For polar_affine and stable_polar_affine, the forces are therefore checked against the
        // expected ones. The other methods can produce incorrect forces on irregular shapes, so their
        // forces are checked against the ones they compute themselves, which bypasses the force check
        // and leaves only the tangent checked.
        for (const auto& rotationMethod : rotationMethods)
        {
            SCOPED_TRACE("rotationMethod = " + rotationMethod);
            this->force->findData("rotationMethod")->read(rotationMethod);

            if (rotationMethod == "polar_affine" || rotationMethod == "stable_polar_affine")
            {
                this->run_test(x, v, expectedForces);
            }
            else
            {
                this->run_test(x, v, forcesAt(x, v));
            }
        }
    }
};

using TestTypes = ::testing::Types<
    CorotationalFEMForceField<Vec2Types, sofa::geometry::Quad>,
    CorotationalFEMForceField<Vec2Types, sofa::geometry::Triangle>,
    CorotationalFEMForceField<Vec3Types, sofa::geometry::Hexahedron>,
    CorotationalFEMForceField<Vec3Types, sofa::geometry::Tetrahedron>,
    CorotationalFEMForceField<Vec2Types, sofa::geometry::Edge>,
    CorotationalFEMForceField<Vec3Types, sofa::geometry::Quad>,
    CorotationalFEMForceField<Vec3Types, sofa::geometry::Triangle>>;

TYPED_TEST_SUITE(CorotationalFEMForceFieldTangent_test, TestTypes);

TYPED_TEST(CorotationalFEMForceFieldTangent_test, rotated)
{
    using Matrix = Matrix<typename TestFixture::DataTypes>;
    this->runTest(this->rotation, Matrix::Identity());
}

TYPED_TEST(CorotationalFEMForceFieldTangent_test, stretched)
{
    using Matrix = Matrix<typename TestFixture::DataTypes>;
    this->runTest(Matrix::Identity(), this->stretch);
}

TYPED_TEST(CorotationalFEMForceFieldTangent_test, stretchedRotated)
{
    this->runTest(this->rotation, this->stretch);
}

}
