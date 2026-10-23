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

struct PolarMethod { static constexpr const char* key = "polar"; };
struct PolarAffineMethod { static constexpr const char* key = "polar_affine"; };
struct StablePolarMethod { static constexpr const char* key = "stable_polar"; };
struct StablePolarAffineMethod { static constexpr const char* key = "stable_polar_affine"; };
struct TriangleMethod { static constexpr const char* key = "triangle"; };
struct HexahedronMethod { static constexpr const char* key = "hexahedron"; };

using Quad2D = CorotationalFEMForceField<Vec2Types, sofa::geometry::Quad>;
using Triangle2D = CorotationalFEMForceField<Vec2Types, sofa::geometry::Triangle>;
using Edge2D = CorotationalFEMForceField<Vec2Types, sofa::geometry::Edge>;
using Quad3D = CorotationalFEMForceField<Vec3Types, sofa::geometry::Quad>;
using Triangle3D = CorotationalFEMForceField<Vec3Types, sofa::geometry::Triangle>;
using Hexahedron3D = CorotationalFEMForceField<Vec3Types, sofa::geometry::Hexahedron>;
using Tetrahedron3D = CorotationalFEMForceField<Vec3Types, sofa::geometry::Tetrahedron>;

template <class ForceFieldType, class MethodType>
struct TestCase
{
    using ForceField = ForceFieldType;
    using Method = MethodType;
};

}

/// Checks the force derivatives against a finite difference of the force, on a single element.
template <class TestCase>
struct CorotationalFEMForceFieldTangent_test : public sofa::ForceField_test<typename TestCase::ForceField>
{
    using ForceField = typename TestCase::ForceField;
    using DataTypes = typename ForceField::DataTypes;
    using VecCoord = sofa::VecCoord_t<DataTypes>;
    using VecDeriv = sofa::VecDeriv_t<DataTypes>;
    using Coord = sofa::Coord_t<DataTypes>;

    /// Rest positions of the element. 
    VecCoord restPositions;

    Matrix<DataTypes> rotation;
    Matrix<DataTypes> stretch;

    CorotationalFEMForceFieldTangent_test()
    {
        const auto topology = sofa::core::objectmodel::New<sofa::component::topology::container::constant::MeshTopology>();
        this->node->addObject(topology);

        /// An irregular shape is chosen for all elements that follow. This skews the rigid fit of
        /// the polar decomposition methods. Rigid and affine fits extract different rotations and
        /// the tangent of each must follow how its own rotation changes.
        if constexpr (std::is_same_v<ForceField, Quad2D>)
        {
            topology->addQuad(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(3.0, 0.0), Coord(3.0, 1.0), Coord(0.0, 1.0)};
            rotation = prescribedRotationMatrix2D;
            stretch = prescribedStretchMatrix2D;
        }
        else if constexpr (std::is_same_v<ForceField, Triangle2D>)
        {
            topology->addTriangle(0, 1, 2);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(1.0, 0.0), Coord(0.0, 1.0)};
            rotation = prescribedRotationMatrix2D;
            stretch = prescribedStretchMatrix2D;
        }
        else if constexpr (std::is_same_v<ForceField, Edge2D>)
        {
            topology->addEdge(0, 1);
            restPositions = VecCoord{Coord(0.0, 0.0), Coord(1.0, 0.0)};
            rotation = prescribedRotationMatrix2D;
            stretch = prescribedEdgeStretchMatrix2D;
        }
        else if constexpr (std::is_same_v<ForceField, Quad3D>)
        {
            topology->addQuad(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(3.0, 0.0, 0.0), Coord(3.0, 1.0, 0.0), Coord(0.0, 1.0, 0.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedFlatStretchMatrix3D;
        }
        else if constexpr (std::is_same_v<ForceField, Triangle3D>)
        {
            topology->addTriangle(0, 1, 2);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(1.0, 0.0, 0.0), Coord(0.0, 1.0, 0.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedFlatStretchMatrix3D;
        }
        else if constexpr (std::is_same_v<ForceField, Hexahedron3D>)
        {
            topology->addHexa(0, 1, 2, 3, 4, 5, 6, 7);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(3.0, 0.0, 0.0), Coord(3.0, 2.0, 0.0), Coord(0.0, 2.0, 0.0),
                                     Coord(0.0, 0.0, 1.0), Coord(3.0, 0.0, 1.0), Coord(3.0, 2.0, 1.0), Coord(0.0, 2.0, 1.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedStretchMatrix3D;
        }
        else if constexpr (std::is_same_v<ForceField, Tetrahedron3D>)
        {
            topology->addTetra(0, 1, 2, 3);
            restPositions = VecCoord{Coord(0.0, 0.0, 0.0), Coord(1.0, 0.0, 0.0), Coord(0.0, 1.0, 0.0), Coord(0.0, 0.0, 1.0)};
            rotation = prescribedRotationMatrix3D;
            stretch = prescribedStretchMatrix3D;
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
    VecDeriv computeForcesAt(const VecCoord& x, const VecDeriv& v)
    {
        this->setupState(x, v);
        sofa::simulation::node::initRoot(this->node.get());
        core::MechanicalParams mparams;
        this->computeForce(&mparams);
        VecDeriv forces;
        sofa::testing::copyFromData(forces, this->dof->readForces());
        return forces;
    }

    /// Moves the nodes by the deformation gradient F = R U and runs the ForceField test there.
    /// The consistency of the tangent stiffness is checked therein.
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
        VecDeriv expectedForces = computeForcesAt(stretchedPositions, v);
        for (auto& force : expectedForces)
        {
            force = R * force;
        }

        using Method = typename TestCase::Method;
        this->force->findData("rotationMethod")->read(Method::key);

        // For irregular elements, only the polar decomposition of the affine fit extracts the
        // rotation that produces the correct forces. 
        if constexpr (std::is_same_v<Method, PolarAffineMethod> || std::is_same_v<Method, StablePolarAffineMethod>)
        {
            // Therefore the expected force is used as reference.
            this->run_test(x, v, expectedForces);
        }
        else
        {
            // The rest of the methods produce incorrect forces on irregular shapes. So the check is
            // implicitly bypassed by using the force they compute themselves as reference. These
            // methods are also expected to miscompute the potential energy on irregular shapes.
            this->flags &= ~sofa::ForceField_test<ForceField>::TEST_POTENTIAL_ENERGY;
            this->run_test(x, v, computeForcesAt(x, v));
        }
    }
};

using TestTypes = ::testing::Types<
    TestCase<Quad2D, PolarMethod>,
    TestCase<Quad2D, PolarAffineMethod>,
    TestCase<Quad2D, StablePolarMethod>,
    TestCase<Quad2D, StablePolarAffineMethod>,
    TestCase<Triangle2D, PolarMethod>,
    TestCase<Triangle2D, PolarAffineMethod>,
    TestCase<Triangle2D, StablePolarMethod>,
    TestCase<Triangle2D, StablePolarAffineMethod>,
    TestCase<Edge2D, PolarMethod>,
    TestCase<Edge2D, PolarAffineMethod>,
    TestCase<Edge2D, StablePolarMethod>,
    TestCase<Edge2D, StablePolarAffineMethod>,
    TestCase<Quad3D, PolarMethod>,
    TestCase<Quad3D, PolarAffineMethod>,
    TestCase<Quad3D, StablePolarMethod>,
    TestCase<Quad3D, StablePolarAffineMethod>,
    TestCase<Triangle3D, PolarMethod>,
    TestCase<Triangle3D, PolarAffineMethod>,
    TestCase<Triangle3D, StablePolarMethod>,
    TestCase<Triangle3D, StablePolarAffineMethod>,
    TestCase<Triangle3D, TriangleMethod>,
    TestCase<Hexahedron3D, PolarMethod>,
    TestCase<Hexahedron3D, PolarAffineMethod>,
    TestCase<Hexahedron3D, StablePolarMethod>,
    TestCase<Hexahedron3D, StablePolarAffineMethod>,
    TestCase<Hexahedron3D, HexahedronMethod>,
    TestCase<Tetrahedron3D, PolarMethod>,
    TestCase<Tetrahedron3D, PolarAffineMethod>,
    TestCase<Tetrahedron3D, StablePolarMethod>,
    TestCase<Tetrahedron3D, StablePolarAffineMethod>,
    TestCase<Tetrahedron3D, TriangleMethod>>;

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
