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
 * residual-based stopping criterion (d_residueThreshold) and no per-iteration history exposed:
 * only the final residual (d_currentResidue) can be inspected. So only the scenario that maps
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
        return std::sqrt(staticSolver->d_currentResidue.getValue());
    }


    NodeSPtr root;
    StaticEquilibriumIntegrationScheme::SPtr staticSolver;
};

TEST_F(StaticEquilibriumIntegrationSchemeTest, Residuals) {
    // This is the expected force residual norm after 10 Newton iterations, with the convergence
    // threshold disabled so the solve always runs the maximum number of iterations.
    const SReal expected_force_residual_norm = 1.2283305451908398e-05;

    staticSolver->d_maxNbIterationsNewton.setValue(10);
    staticSolver->d_residueThreshold.setValue(0.0_sreal);

    const SReal actual_force_residual_norm = this->execute();

    EXPECT_NEAR(actual_force_residual_norm, expected_force_residual_norm, 1e-3)
    << "The actual force residual norm doesn't match the expected one.";
}
