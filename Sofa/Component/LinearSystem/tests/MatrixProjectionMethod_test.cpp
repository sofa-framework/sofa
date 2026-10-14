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

/// The container of the global matrix, the linear system and the projection method under
/// test. The linear system and the projection method both have a variant taking advantage
/// of a constant sparsity pattern, and both variants take a different code path once the
/// pattern has been built, so every combination is tested.
struct Parameters
{
    std::string matrixTemplate;
    std::string linearSystem;
    std::string projectionMethod;
};

static std::string testName(const ::testing::TestParamInfo<Parameters>& info)
{
    return info.param.matrixTemplate + "_" + info.param.linearSystem + "_"
        + info.param.projectionMethod;
}

/// The assembled matrix is 6x6: two independent particles of 3 degrees of freedom.
static constexpr MatrixIndex matrixSize = 6;

/// The number of times the matrix is assembled. The systems and projection methods
/// relying on a constant sparsity pattern only use it from the second assembly on, so
/// more than one is required to cover them.
static constexpr unsigned int nbAssembly = 3;

/// Assembles the matrix of a scene made of two independent particles coupled by a
/// spring. The spring is applied either directly on the degrees of freedom, or on
/// identity-mapped copies of them:
///
///     root
///      |- <linear system>
///      |- p0 - dofs               (+ p0/mapped/dofs and an IdentityMapping)
///      |- p1 - dofs               (+ p1/mapped/dofs and an IdentityMapping)
///      |- SpringForceField        (on the dofs, or on the mapped ones)
///      |- <projection methods>    (only when the spring is applied on mapped states)
///
/// The matrix is assembled `nbAssembly` times, and every assembly is returned.
static void assembleMatrix(const Parameters& parameters, bool throughMappings,
                           std::array<Matrix, nbAssembly>& results)
{
    const auto plugins = testing::makeScopedPlugin({
        Sofa.Component.LinearSystem,
        Sofa.Component.Mapping.Linear,
        Sofa.Component.SolidMechanics.Spring,
        Sofa.Component.StateContainer});

    const simulation::Node::SPtr root = simulation::getSimulation()->createNewGraph("root");

    const auto linearSystem = simpleapi::createObject(root, parameters.linearSystem,
        {{"template", parameters.matrixTemplate}});

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

    if (throughMappings)
    {
        // one projection method per ordered pair of mapped states: the contribution of
        // the spring is split into as many mapped matrices. Contrary to the global
        // matrix, the mapped matrices are always in the CRS format.
        for (const auto& first : springTargets)
        {
            for (const auto& second : springTargets)
            {
                simpleapi::createObject(root, parameters.projectionMethod,
                    {{"template", "CompressedRowSparseMatrixd"},
                     {"mechanicalStates", first + " " + second}});
            }
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

    auto* forceField = dynamic_cast<core::behavior::BaseForceField*>(spring.get());
    ASSERT_NE(forceField, nullptr);

    auto* system = dynamic_cast<core::behavior::BaseMatrixLinearSystem*>(linearSystem.get());
    ASSERT_NE(system, nullptr);

    for (unsigned int assembly = 0; assembly < nbAssembly; ++assembly)
    {
        // force fields usually pre-compute elements required by the assembly in addForce
        core::MultiVecDerivId forceId = core::vec_id::write_access::externalForce;
        forceField->addForce(&mparams, forceId);

        system->buildSystemMatrix(&mparams);

        const linearalgebra::BaseMatrix* matrix = system->getSystemBaseMatrix();
        ASSERT_NE(matrix, nullptr);
        ASSERT_EQ(matrix->rowSize(), matrixSize);
        ASSERT_EQ(matrix->colSize(), matrixSize);

        Matrix& result = results[assembly];
        result.resize(matrixSize, matrixSize);
        for (MatrixIndex i = 0; i < matrixSize; ++i)
        {
            for (MatrixIndex j = 0; j < matrixSize; ++j)
            {
                result.set(i, j, matrix->element(i, j));
            }
        }
    }

    simulation::node::unload(root);
}

class MatrixProjectionMethodTest : public ::testing::TestWithParam<Parameters> {};

/// A force field acting on mapped states is projected into the global matrix as
/// J^T K J. The mappings here are identities, so J = I, and the projection must
/// reproduce exactly the matrix obtained by applying the same force field directly on
/// the degrees of freedom of the system.
///
/// The comparison is sensitive in both directions: a missing coupling term and a
/// spurious contribution both break the equality.
TEST_P(MatrixProjectionMethodTest, mappedForceFieldMatchesNonMapped)
{
    std::array<Matrix, nbAssembly> reference, projected;
    assembleMatrix(GetParam(), false, reference);
    assembleMatrix(GetParam(), true, projected);

    static constexpr SReal tolerance = 1e-12_sreal;

    for (unsigned int assembly = 0; assembly < nbAssembly; ++assembly)
    {
        for (MatrixIndex i = 0; i < matrixSize; ++i)
        {
            for (MatrixIndex j = 0; j < matrixSize; ++j)
            {
                EXPECT_NEAR(projected[assembly].element(i, j), reference[assembly].element(i, j), tolerance)
                    << "at (" << i << ", " << j << ") of assembly " << assembly;
            }
        }
    }
}

/// Guards the test above: it would still pass if both matrices were empty. The spring
/// couples the two particles, so both off-diagonal blocks must be non-zero, i.e. the
/// projection must produce the coupling terms and not only the diagonal ones.
TEST_P(MatrixProjectionMethodTest, mappedForceFieldProducesCouplingTerms)
{
    std::array<Matrix, nbAssembly> projected;
    assembleMatrix(GetParam(), true, projected);

    for (unsigned int assembly = 0; assembly < nbAssembly; ++assembly)
    {
        SReal diagonalBlocks = 0_sreal;
        SReal offDiagonalBlocks = 0_sreal;

        for (MatrixIndex i = 0; i < matrixSize; ++i)
        {
            for (MatrixIndex j = 0; j < matrixSize; ++j)
            {
                const bool sameParticle = (i < 3) == (j < 3);
                (sameParticle ? diagonalBlocks : offDiagonalBlocks) +=
                    std::abs(projected[assembly].element(i, j));
            }
        }

        EXPECT_GT(diagonalBlocks, 0_sreal) << "at assembly " << assembly;
        EXPECT_GT(offDiagonalBlocks, 0_sreal) << "at assembly " << assembly;
    }
}

INSTANTIATE_TEST_SUITE_P(MatrixProjectionMethod, MatrixProjectionMethodTest,
    ::testing::ValuesIn(std::vector<Parameters>{
        {"FullMatrix", "MatrixLinearSystem", "MatrixProjectionMethod"},
        {"FullMatrix", "MatrixLinearSystem", "ConstantSparsityProjectionMethod"},
        {"CompressedRowSparseMatrixd", "MatrixLinearSystem", "MatrixProjectionMethod"},
        {"CompressedRowSparseMatrixd", "MatrixLinearSystem", "ConstantSparsityProjectionMethod"},
        // ConstantSparsityPatternSystem only exists for the CRS format
        {"CompressedRowSparseMatrixd", "ConstantSparsityPatternSystem", "MatrixProjectionMethod"},
        {"CompressedRowSparseMatrixd", "ConstantSparsityPatternSystem", "ConstantSparsityProjectionMethod"},
    }),
    testName);

}
