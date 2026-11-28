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
#include <sofa/simulation/Node.h>
#include <sofa/component/integrationscheme/backward/StaticEquilibriumIntegrationScheme.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/State.h>
#include <sofa/defaulttype/VecTypes.h>

#include <vector>
#include <algorithm>

using namespace sofa::simulation;
using namespace sofa::simpleapi;
using namespace sofa::helper::logging;

using sofa::simulation::graph::DAGSimulation;
using sofa::component::integrationscheme::backward::StaticEquilibriumIntegrationScheme;

static constexpr SReal poissonRatio = 0;
static constexpr SReal youngModulus = 3000;
static constexpr SReal mu = youngModulus / (2.0 * (1.0 + poissonRatio));
static constexpr SReal l = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

/**
 * Create a bending rectangular beam simulation using the StaticEquilibriumIntegrationScheme.
 *
 * Domain: 15x15x80
 * Discretization: 3x3x9 nodes, linear tetrahedral mesh
 * Material: StVenantKirchhoff (Young modulus = 3000, Poisson ratio = 0.4)
 *
 * The Newton-Raphson iteration used to be a separate NewtonRaphsonSolver component linked to
 * StaticSolver, with a choice of pluggable convergence measures (relative-to-initial,
 * estimate-difference, ...) and a per-iteration residual history (d_residualGraph). After the
 * refactoring, that logic lives directly in StaticEquilibriumIntegrationScheme, with a single
 * residual-based stopping criterion (d_residualThreshold) and no per-iteration history exposed:
 * only the final residual (d_currentResidual) can be inspected. So only the scenario that maps
 * cleanly onto the new API is kept here (force exactly d_maxNbIterationsNewton iterations and
 * check the final residual); the former RelativeResiduals/AbsoluteResiduals cases exercised
 * convergence criteria that no longer exist.
 */
class StaticEquilibriumIntegrationSchemeTest : public sofa::testing::BaseTest
{
public:
    void doSetUp() override {

        root = getSimulation()->createNewNode("root");

        this->loadPlugins({
            Sofa.Component.Topology.Container.Grid,
            Sofa.Component.IntegrationScheme.Backward,
            Sofa.Component.LinearSolver.Direct,
            Sofa.Component.StateContainer,
            Sofa.Component.Topology.Container.Dynamic,
            Sofa.Component.Topology.Mapping,
            Sofa.Component.SolidMechanics.FEM.HyperElastic,
            Sofa.Component.Engine.Select,
            Sofa.Component.Constraint.Projective,
            Sofa.Component.MechanicalLoad
        });

        createObject(root, "DefaultAnimationLoop");
        createObject(root, "RegularGridTopology", {{"name", "grid"}, {"min", "-7.5 -7.5 0"}, {"max", "7.5 7.5 80"}, {"n", "3 3 9"}});
        const auto s = createObject(root, "StaticEquilibriumIntegrationScheme", {{"maxNbIterationsNewton", "10"}});
        createObject(root, "SparseLDLSolver", {{"template", "CompressedRowSparseMatrixd"}});
        createObject(root, "MechanicalObject", {{"name", "mo"}, {"src", "@grid"}});
        createObject(root, "TetrahedronSetTopologyContainer", {{"name", "mechanical_topology"}});
        createObject(root, "TetrahedronSetTopologyModifier");
        createObject(root, "Hexa2TetraTopologicalMapping", {{"input", "@grid"}, {"output", "@mechanical_topology"}});
        createObject(root, "TetrahedronHyperelasticityFEMForceField", {
            {"name", "FEM"},
            {"materialName", "StVenantKirchhoff"},
            {"ParameterSet", std::to_string(mu) + " " + std::to_string(l)},
            {"topology", "@mechanical_topology"}
        });
        ASSERT_NE(root->getObject("FEM"), nullptr);
        ASSERT_NE(root->getObject("FEM")->findData("materialName"), nullptr);
        ASSERT_EQ(root->getObject("FEM")->findData("materialName")->getValueString(), "StVenantKirchhoff");

        createObject(root, "BoxROI", {{"name", "top_roi"}, {"box", "-7.5 -7.5 -0.9 7.5 7.5 0.1"}, {"triangles", "@mechanical_topology.triangles"}});
        createObject(root, "FixedProjectiveConstraint", {{"indices", "@top_roi.indices"}});

        createObject(root, "BoxROI", {{"name", "base_roi"}, {"box", "-7.5 -7.5 79.9 7.5 7.5 80.1"}, {"triangles", "@mechanical_topology.triangles"}});
        createObject(root, "SurfacePressureForceField", {{"pressure", "100"}, {"mainDirection", "0 -1 0"}, {"triangleIndices", "@base_roi.trianglesInROI"}});

        staticSolver = dynamic_cast<StaticEquilibriumIntegrationScheme *> (s.get());
    }

    void doTearDown() override {
        sofa::simulation::node::unload(root);
    }

    /// Runs one time step and returns the norm of the force residual after the Newton loop stopped.
    auto execute() -> SReal {
        sofa::simulation::node::initRoot(root.get());
        sofa::simulation::node::animate(root.get(), 1_sreal);
        return std::sqrt(staticSolver->d_currentResidual.getValue());
    }


    NodeSPtr root;
    StaticEquilibriumIntegrationScheme::SPtr staticSolver;
};

TEST_F(StaticEquilibriumIntegrationSchemeTest, Residuals) {
    // This is the expected force residual norm after 10 Newton iterations, with the convergence
    // threshold disabled so the solve always runs the maximum number of iterations.
    const SReal expected_force_residual_norm = 1.2283305451908398e-05;

    staticSolver->d_maxNbIterationsNewton.setValue(10);
    staticSolver->d_residualThreshold.setValue(0.0_sreal);

    const SReal actual_force_residual_norm = this->execute();

    EXPECT_NEAR(actual_force_residual_norm, expected_force_residual_norm, 1e-3)
    << "The actual force residual norm doesn't match the expected one.";
}

/**
 * The integration scheme must solve the static equilibrium on the state vectors it is given
 * (xResult, vResult), which are not necessarily position and velocity. For example,
 * FreeMotionAnimationLoop calls it with freePosition and freeVelocity.
 *
 * Forces and residual must be evaluated on xResult. If they were evaluated on position instead,
 * the residual would never change along the Newton iterations, and xResult would not converge
 * toward the equilibrium.
 *
 * The result obtained in freePosition is compared to the result obtained in position.
 */
TEST_F(StaticEquilibriumIntegrationSchemeTest, ResultInFreePosition)
{
    using DataTypes = sofa::defaulttype::Vec3Types;
    using VecCoord = DataTypes::VecCoord;
    using VecDeriv = DataTypes::VecDeriv;

    staticSolver->d_maxNbIterationsNewton.setValue(10);
    staticSolver->d_residualThreshold.setValue(0.0_sreal);

    sofa::simulation::node::initRoot(root.get());

    auto* state = dynamic_cast<sofa::core::State<DataTypes>*>(root->getObject("mo"));
    ASSERT_NE(state, nullptr);

    const VecCoord initialPosition = state->read(sofa::core::vec_id::read_access::position)->getValue();
    const VecDeriv initialVelocity = state->read(sofa::core::vec_id::read_access::velocity)->getValue();
    ASSERT_FALSE(initialPosition.empty());

    // The free state starts from the current state, as in FreeMotionAnimationLoop
    state->write(sofa::core::vec_id::write_access::freePosition)->setValue(initialPosition);
    state->write(sofa::core::vec_id::write_access::freeVelocity)->setValue(initialVelocity);

    const auto* params = sofa::core::ExecParams::defaultInstance();

    // Solve the static equilibrium in freePosition
    staticSolver->integrate(params, 1_sreal,
        sofa::core::vec_id::write_access::freePosition,
        sofa::core::vec_id::write_access::freeVelocity);
    const SReal residualInFreePosition = std::sqrt(staticSolver->d_currentResidual.getValue());

    const VecCoord freePosition = state->read(sofa::core::vec_id::read_access::freePosition)->getValue();

    // position must not have been modified
    {
        const VecCoord& position = state->read(sofa::core::vec_id::read_access::position)->getValue();
        ASSERT_EQ(position.size(), initialPosition.size());
        for (std::size_t i = 0; i < position.size(); ++i)
        {
            EXPECT_EQ(position[i], initialPosition[i]) << "position has been modified at node " << i;
        }
    }

    // freePosition must have moved toward the equilibrium
    SReal maxDisplacement = 0;
    for (std::size_t i = 0; i < freePosition.size(); ++i)
    {
        maxDisplacement = std::max(maxDisplacement, (freePosition[i] - initialPosition[i]).norm());
    }
    EXPECT_GT(maxDisplacement, 1_sreal) << "freePosition did not move";

    // Reference: solve the same static equilibrium in position
    staticSolver->integrate(params, 1_sreal,
        sofa::core::vec_id::write_access::position,
        sofa::core::vec_id::write_access::velocity);
    const SReal residualInPosition = std::sqrt(staticSolver->d_currentResidual.getValue());

    const VecCoord& position = state->read(sofa::core::vec_id::read_access::position)->getValue();

    EXPECT_NEAR(residualInFreePosition, residualInPosition, 1e-10);
    EXPECT_LT(residualInFreePosition, 1e-3) << "The Newton iterations did not converge in freePosition";

    ASSERT_EQ(position.size(), freePosition.size());
    for (std::size_t i = 0; i < position.size(); ++i)
    {
        for (std::size_t c = 0; c < DataTypes::spatial_dimensions; ++c)
        {
            EXPECT_NEAR(freePosition[i][c], position[i][c], 1e-8) << "node " << i << ", coordinate " << c;
        }
    }
}
