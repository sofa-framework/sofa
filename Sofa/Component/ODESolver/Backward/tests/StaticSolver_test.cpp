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
#include <sofa/component/odesolver/backward/StaticSolver.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simpleapi/SimpleApi.h>

#include <vector>
#include <algorithm>

using namespace sofa::simulation;
using namespace sofa::simpleapi;
using namespace sofa::helper::logging;

using sofa::simulation::graph::DAGSimulation;
using sofa::component::odesolver::backward::StaticSolver;

static constexpr SReal poissonRatio = 0;
static constexpr SReal youngModulus = 3000;
static constexpr SReal mu = youngModulus / (2.0 * (1.0 + poissonRatio));
static constexpr SReal l = youngModulus * poissonRatio / ((1.0 + poissonRatio) * (1.0 - 2.0 * poissonRatio));

/**
 * Create a bending rectangular beam simulation using the StaticSolver.
 *
 * Domain: 15x15x80
 * Discretization: 3x3x9 nodes, linear tetrahedral mesh
 * Material: StVenantKirchhoff (Young modulus = 3000, Poisson ratio = 0.4)
 *
 */
class StaticSolverTest : public sofa::testing::BaseTest
{
public:
    void doSetUp() override {

        root = getSimulation()->createNewNode("root");

        this->loadPlugins({
            Sofa.Component.Topology.Container.Grid,
            Sofa.Component.ODESolver.Backward,
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
        const auto s = createObject(root, "StaticSolver");
        const auto n = createObject(root, "NewtonRaphsonSolver", {{"maxNbIterationsNewton", "10"}});
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

        staticSolver = dynamic_cast<StaticSolver *> (s.get());
        newtonSolver = dynamic_cast<sofa::component::odesolver::backward::NewtonRaphsonSolver*> (n.get());
    }

    void doTearDown() override {
        sofa::simulation::node::unload(root);
    }

    auto execute() -> std::vector<SReal> {
        using namespace std;
        sofa::simulation::node::initRoot(root.get());
        sofa::simulation::node::animate(root.get(), 1_sreal);
        const auto& residualGraph = newtonSolver->d_residualGraph.getValue();
        auto residuals = residualGraph.at("residual");
        transform(begin(residuals), end(residuals), begin(residuals), [](SReal r) {return sqrt(r);});
        return residuals;
    }


    NodeSPtr root;
    StaticSolver::SPtr staticSolver;
    sofa::component::odesolver::backward::NewtonRaphsonSolver::SPtr newtonSolver;
};

TEST_F(StaticSolverTest, Residuals) {
    // These are the expected force residual if we do not activate any convergence threshold
    // and force the solve to do 10 Newton iterations
    const std::vector<SReal> expected_force_residual_norms = {
        6286.9334449829712,
        2535.2291278587031,
        196.33640050006048,
        35.551907341224727,
        2.4346076657342617,
        0.25822327664821382,
        0.028069066914505583,
        0.0041872264223336737,
        0.00058185560326781769,
        8.5402692438165759e-05,
        1.2283305451908398e-05
    };

    // Disable all convergence criteria
    newtonSolver->d_maxNbIterationsNewton.setValue(10);
    newtonSolver->d_relativeSuccessiveStoppingThreshold.setValue(-1);
    newtonSolver->d_relativeInitialStoppingThreshold.setValue(-1);
    newtonSolver->d_absoluteResidualStoppingThreshold.setValue(-1);
    newtonSolver->d_relativeEstimateDifferenceThreshold.setValue(-1);
    newtonSolver->d_absoluteEstimateDifferenceThreshold.setValue(-1);
    newtonSolver->d_warnWhenDiverge.setValue(false);

    const std::vector<SReal> actual_force_residual_norms = this->execute();

    EXPECT_EQ(actual_force_residual_norms.size(), expected_force_residual_norms.size())
    << "The static ODE solver is supposed to execute 10 Newton steps since the convergence criteria were deactivated.";

    for (std::size_t newton_it = 0; newton_it < actual_force_residual_norms.size(); ++newton_it) {
        EXPECT_NEAR(actual_force_residual_norms[newton_it], expected_force_residual_norms[newton_it], 1e-3)
        << "The actual force residual norm doesn't match the expected one.";
    }
}

TEST_F(StaticSolverTest, RelativeResiduals)
{
    // Disable all convergence criteria BUT the relative residual

    newtonSolver->d_maxNbIterationsNewton.setValue(10);
    newtonSolver->d_relativeSuccessiveStoppingThreshold.setValue(-1);
    newtonSolver->d_relativeInitialStoppingThreshold.setValue(1e-5_sreal);
    newtonSolver->d_absoluteResidualStoppingThreshold.setValue(-1);
    newtonSolver->d_relativeEstimateDifferenceThreshold.setValue(1e-5_sreal);
    newtonSolver->d_absoluteEstimateDifferenceThreshold.setValue(-1);
    newtonSolver->d_warnWhenDiverge.setValue(false);

    const sofa::type::vector<SReal> actual_force_residual_norms = this->execute();
    EXPECT_EQ(actual_force_residual_norms.size(), 7)
    << "The static ODE solver is supposed to converge after 7 Newton steps when using a relative residual threshold of 1e-5.\n"
    << actual_force_residual_norms;
}

TEST_F(StaticSolverTest, AbsoluteResiduals) {
    // Disable all convergence criteria BUT the absolute residual
    newtonSolver->d_maxNbIterationsNewton.setValue(10);
    newtonSolver->d_relativeSuccessiveStoppingThreshold.setValue(-1);
    newtonSolver->d_relativeInitialStoppingThreshold.setValue(-1);
    newtonSolver->d_absoluteResidualStoppingThreshold.setValue(-1);
    newtonSolver->d_relativeEstimateDifferenceThreshold.setValue(-1);
    newtonSolver->d_absoluteEstimateDifferenceThreshold.setValue(1e-5_sreal);
    newtonSolver->d_warnWhenDiverge.setValue(true);

    const sofa::type::vector<SReal> actual_force_residual_norms = this->execute();
    EXPECT_EQ(actual_force_residual_norms.size(), 9)
    << "The static ODE solver is supposed to converge after 9 Newton steps when using an absolute residual threshold of 1e-5.\n"
    << actual_force_residual_norms;
}
