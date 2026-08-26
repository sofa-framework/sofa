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

#include <sofa/component/linearsystem/MatrixLinearSystem.inl>
#include <sofa/component/linearsystem/TypedMatrixLinearSystem.inl>

#include <sofa/component/mapping/linear/IdentityMapping.h>
#include <sofa/component/solidmechanics/spring/SpringForceField.h>
#include <sofa/component/statecontainer/MechanicalObject.h>

#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/ForceField.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>

namespace sofa
{

using DataTypes = defaulttype::Vec3Types;
using MatrixType = linearalgebra::FullMatrix<SReal>;
using VectorType = linearalgebra::FullVector<SReal>;
using MatrixSystem = component::linearsystem::MatrixLinearSystem<MatrixType, VectorType>;
using MechanicalObject3 = component::statecontainer::MechanicalObject<DataTypes>;
using Spring3 = component::solidmechanics::spring::SpringForceField<DataTypes>;
using IdentityMapping3 = component::mapping::linear::IdentityMapping<DataTypes, DataTypes>;

constexpr SReal springStiffness = 100_sreal;
constexpr SReal springRestLength = 1_sreal;

/// A particle, alone in its node, optionally carrying an identity-mapped copy of itself
struct Particle
{
    simulation::Node::SPtr node;
    MechanicalObject3::SPtr state;    ///< the degrees of freedom of the system
    MechanicalObject3::SPtr mapped;   ///< the identity-mapped copy, if any

    /// The state the force field is applied on
    MechanicalObject3* target() const
    {
        return mapped ? mapped.get() : state.get();
    }
};

static Particle createParticle(simulation::Node* parent, const std::string& name,
                               const type::Vec3& position, bool withMapping)
{
    Particle particle;
    particle.node = parent->createChild(name);

    particle.state = core::objectmodel::New<MechanicalObject3>();
    particle.state->setName("dofs");
    particle.node->addObject(particle.state);
    particle.state->resize(1);
    particle.state->writePositions()[0] = position;

    if (withMapping)
    {
        const auto mappedNode = particle.node->createChild(name + "_mapped");

        particle.mapped = core::objectmodel::New<MechanicalObject3>();
        particle.mapped->setName("dofs");
        mappedNode->addObject(particle.mapped);
        particle.mapped->resize(1);
        particle.mapped->writePositions()[0] = position;

        const auto mapping = core::objectmodel::New<IdentityMapping3>();
        mapping->setModels(particle.state.get(), particle.mapped.get());
        mappedNode->addObject(mapping);
    }

    return particle;
}

/// Assembles the matrix of a system made of two independent particles coupled by a
/// spring. When `throughMappings` is true, the spring is applied on identity-mapped
/// copies of the particles rather than on the particles themselves. In both cases the
/// degrees of freedom of the system are the two particles, so the matrix is 6x6.
static void assembleMatrix(bool throughMappings, MatrixType& result)
{
    const simulation::Node::SPtr root = simulation::getSimulation()->createNewGraph("root");

    const MatrixSystem::SPtr linearSystem = core::objectmodel::New<MatrixSystem>();
    root->addObject(linearSystem);

    const Particle particle0 = createParticle(root.get(), "p0", {0_sreal, 0_sreal, 0_sreal}, throughMappings);
    const Particle particle1 = createParticle(root.get(), "p1", {2_sreal, 0_sreal, 0_sreal}, throughMappings);

    const auto spring = core::objectmodel::New<Spring3>(particle0.target(), particle1.target());
    spring->setName("spring");
    root->addObject(spring);
    spring->addSpring(0, 0, springStiffness, 0_sreal, springRestLength);

    simulation::node::initRoot(root.get());

    auto mparams = *core::MechanicalParams::defaultInstance();
    mparams.setKFactor(1_sreal);

    // force fields usually pre-compute elements required by the assembly in addForce
    core::MultiVecDerivId forceId = core::vec_id::write_access::externalForce;
    static_cast<core::behavior::BaseForceField*>(spring.get())->addForce(&mparams, forceId);

    linearSystem->buildSystemMatrix(&mparams);

    const MatrixType* matrix = linearSystem->getSystemMatrix();
    ASSERT_NE(matrix, nullptr);

    result.resize(matrix->rowSize(), matrix->colSize());
    for (MatrixType::Index i = 0; i < matrix->rowSize(); ++i)
    {
        for (MatrixType::Index j = 0; j < matrix->colSize(); ++j)
        {
            result.set(i, j, matrix->element(i, j));
        }
    }

    simulation::node::unload(root);
}

/// A force field applied on mapped states must be projected into the global matrix as
/// J^T K J. Here the mappings are identities, so J = I and the projected matrix must be
/// exactly the matrix obtained by applying the same force field directly on the degrees
/// of freedom of the system.
///
/// The comparison is sensitive in both directions: a missing coupling term and a
/// spurious contribution both break the equality.
TEST(MatrixProjectionMethod, mappedForceFieldMatchesNonMapped)
{
    MatrixType reference, projected;
    assembleMatrix(false, reference);
    assembleMatrix(true, projected);

    // two particles of 3 degrees of freedom each
    ASSERT_EQ(reference.rowSize(), 6);
    ASSERT_EQ(reference.colSize(), 6);
    ASSERT_EQ(projected.rowSize(), reference.rowSize());
    ASSERT_EQ(projected.colSize(), reference.colSize());

    static constexpr SReal tolerance = 1e-12_sreal;

    for (MatrixType::Index i = 0; i < reference.rowSize(); ++i)
    {
        for (MatrixType::Index j = 0; j < reference.colSize(); ++j)
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
    MatrixType projected;
    assembleMatrix(true, projected);

    ASSERT_EQ(projected.rowSize(), 6);
    ASSERT_EQ(projected.colSize(), 6);

    SReal diagonalBlocks = 0_sreal;
    SReal offDiagonalBlocks = 0_sreal;

    for (MatrixType::Index i = 0; i < 6; ++i)
    {
        for (MatrixType::Index j = 0; j < 6; ++j)
        {
            const bool sameParticle = (i < 3) == (j < 3);
            (sameParticle ? diagonalBlocks : offDiagonalBlocks) += std::abs(projected.element(i, j));
        }
    }

    EXPECT_GT(diagonalBlocks, 0_sreal);
    EXPECT_GT(offDiagonalBlocks, 0_sreal);
}

}
