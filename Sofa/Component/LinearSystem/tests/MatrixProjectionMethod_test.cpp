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
#include <sofa/testing/BaseTest.h>

#include <sofa/Modules.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMatrixLinearSystem.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

using Matrix = linearalgebra::FullMatrix<SReal>;
using MatrixIndex = linearalgebra::BaseMatrix::Index;

/// Assembles the matrix of a scene made of two independent particles coupled by a
/// spring. The spring is applied either directly on the degrees of freedom, or on
/// identity-mapped copies of them:
///
///     root
///      |- MatrixLinearSystem
///      |- p0 - dofs               (+ p0/mapped/dofs and an IdentityMapping)
///      |- p1 - dofs               (+ p1/mapped/dofs and an IdentityMapping)
///      |- SpringForceField        (on the dofs, or on the mapped ones)
///
/// In both cases the degrees of freedom of the system are the two particles, so the
/// assembled matrix is 6x6.
static void assembleMatrix(bool throughMappings, Matrix& result)
{
    const auto plugins = testing::makeScopedPlugin({
        Sofa.Component.LinearSystem,
        Sofa.Component.Mapping.Linear,
        Sofa.Component.SolidMechanics.Spring,
        Sofa.Component.StateContainer});

    const simulation::Node::SPtr root = simulation::getSimulation()->createNewGraph("root");

    const auto linearSystem = simpleapi::createObject(root, "MatrixLinearSystem",
        {{"template", "FullMatrix"}});

    static const std::array<std::string, 2> positions { "0 0 0", "2 0 0" };
    std::array<std::string, 2> springTargets;

    for (std::size_t i = 0; i < 2; ++i)
    {
        const std::string name = "p" + std::to_string(i);

        const auto particle = simpleapi::createChild(root, name);
        simpleapi::createObject(particle, "MechanicalObject",
            {{"name", "dofs"}, {"template", "Vec3"}, {"position", positions[i]}});

        springTargets[i] = "@/" + name + "/dofs";

        if (throughMappings)
        {
            const auto mapped = simpleapi::createChild(particle, "mapped");
            simpleapi::createObject(mapped, "MechanicalObject",
                {{"name", "dofs"}, {"template", "Vec3"}, {"position", positions[i]}});
            simpleapi::createObject(mapped, "IdentityMapping", {});

            springTargets[i] = "@/" + name + "/mapped/dofs";
        }
    }

    const auto spring = simpleapi::createObject(root, "SpringForceField",
        {{"name", "spring"},
         {"object1", springTargets[0]},
         {"object2", springTargets[1]},
         // index1 index2 stiffness damping restLength
         {"spring", "0 0 100 0 1"}});

    simulation::node::initRoot(root.get());

    auto mparams = *core::MechanicalParams::defaultInstance();
    mparams.setKFactor(1_sreal);

    // force fields usually pre-compute elements required by the assembly in addForce
    auto* forceField = dynamic_cast<core::behavior::BaseForceField*>(spring.get());
    ASSERT_NE(forceField, nullptr);
    core::MultiVecDerivId forceId = core::vec_id::write_access::externalForce;
    forceField->addForce(&mparams, forceId);

    auto* system = dynamic_cast<core::behavior::BaseMatrixLinearSystem*>(linearSystem.get());
    ASSERT_NE(system, nullptr);
    system->buildSystemMatrix(&mparams);

    const linearalgebra::BaseMatrix* matrix = system->getSystemBaseMatrix();
    ASSERT_NE(matrix, nullptr);

    result.resize(matrix->rowSize(), matrix->colSize());
    for (MatrixIndex i = 0; i < matrix->rowSize(); ++i)
    {
        for (MatrixIndex j = 0; j < matrix->colSize(); ++j)
        {
            result.set(i, j, matrix->element(i, j));
        }
    }

    simulation::node::unload(root);
}

/// A force field acting on mapped states is projected into the global matrix as
/// J^T K J. The mappings here are identities, so J = I, and the projection must
/// reproduce exactly the matrix obtained by applying the same force field directly on
/// the degrees of freedom of the system.
///
/// The comparison is sensitive in both directions: a missing coupling term and a
/// spurious contribution both break the equality.
TEST(MatrixProjectionMethod, mappedForceFieldMatchesNonMapped)
{
    Matrix reference, projected;
    assembleMatrix(false, reference);
    assembleMatrix(true, projected);

    // two particles of 3 degrees of freedom each
    ASSERT_EQ(reference.rowSize(), 6);
    ASSERT_EQ(reference.colSize(), 6);
    ASSERT_EQ(projected.rowSize(), reference.rowSize());
    ASSERT_EQ(projected.colSize(), reference.colSize());

    static constexpr SReal tolerance = 1e-12_sreal;

    for (MatrixIndex i = 0; i < reference.rowSize(); ++i)
    {
        for (MatrixIndex j = 0; j < reference.colSize(); ++j)
        {
            EXPECT_NEAR(projected.element(i, j), reference.element(i, j), tolerance)
                << "at (" << i << ", " << j << ")";
        }
    }
}

/// Guards the test above: it would still pass if both matrices were empty. The spring
/// couples the two particles, so both off-diagonal blocks must be non-zero, i.e. the
/// projection must produce the coupling terms and not only the diagonal ones.
TEST(MatrixProjectionMethod, mappedForceFieldProducesCouplingTerms)
{
    Matrix projected;
    assembleMatrix(true, projected);

    ASSERT_EQ(projected.rowSize(), 6);
    ASSERT_EQ(projected.colSize(), 6);

    SReal diagonalBlocks = 0_sreal;
    SReal offDiagonalBlocks = 0_sreal;

    for (MatrixIndex i = 0; i < 6; ++i)
    {
        for (MatrixIndex j = 0; j < 6; ++j)
        {
            const bool sameParticle = (i < 3) == (j < 3);
            (sameParticle ? diagonalBlocks : offDiagonalBlocks) += std::abs(projected.element(i, j));
        }
    }

    EXPECT_GT(diagonalBlocks, 0_sreal);
    EXPECT_GT(offDiagonalBlocks, 0_sreal);
}

}
