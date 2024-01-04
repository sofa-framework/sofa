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
#include <sofa/simulation/graph/SimpleApi.h>

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
    void onSetUp() override {

        root = getSimulation()->createNewNode("root");

        createObject(root, "RequiredPlugin", {{"pluginName", "Sofa.Component"}});
        createObject(root, "DefaultAnimationLoop");
        createObject(root, "RegularGridTopology", {{"name", "grid"}, {"min", "-7.5 -7.5 0"}, {"max", "7.5 7.5 80"}, {"n", "3 3 9"}});
        const auto s = createObject(root, "StaticSolver", {{"newton_iterations", "10"}});
        createObject(root, "SparseLDLSolver", {{"template", "CompressedRowSparseMatrixd"}});
        createObject(root, "MechanicalObject", {{"name", "mo"}, {"src", "@grid"}});
        createObject(root, "TetrahedronSetTopologyContainer", {{"name", "mechanical_topology"}});
        createObject(root, "TetrahedronSetTopologyModifier");
        createObject(root, "Hexa2TetraTopologicalMapping", {{"input", "@grid"}, {"output", "@mechanical_topology"}});
        createObject(root, "TetrahedronHyperelasticityFEMForceField", {
            {"materialName", "StVenantKirchhoff"},
            {"ParameterSet", std::to_string(mu) + " " + std::to_string(l)},
            {"topology", "@mechanical_topology"}
        });

        createObject(root, "BoxROI", {{"name", "top_roi"}, {"box", "-7.5 -7.5 -0.9 7.5 7.5 0.1"}});
        createObject(root, "FixedProjectiveConstraint", {{"indices", "@top_roi.indices"}});

        createObject(root, "BoxROI", {{"name", "base_roi"}, {"box", "-7.5 -7.5 79.9 7.5 7.5 80.1"}});
        createObject(root, "SurfacePressureForceField", {{"pressure", "100"}, {"mainDirection", "0 -1 0"}, {"triangleIndices", "@base_roi.trianglesInROI"}});

        solver = dynamic_cast<StaticSolver *> (s.get());
    }

    void onTearDown() override {
        sofa::simulation::node::unload(root);
    }

    auto execute() -> std::pair<std::vector<SReal>, std::vector<SReal>> {
        using namespace std;
        sofa::simulation::node::initRoot(root.get());
        sofa::simulation::node::animate(root.get(), 1_sreal);
        auto residuals = solver->squared_residual_norms();
        auto corrections = solver->squared_increment_norms();
        transform(begin(residuals), end(residuals), begin(residuals), [](SReal r) {return sqrt(r);});
        transform(begin(corrections), end(corrections), begin(corrections), [](SReal r) {return sqrt(r);});
        return {residuals, corrections};
    }


    NodeSPtr root;
    StaticSolver::SPtr solver;
};

TEST_F(StaticSolverTest, Residuals) {
    using namespace sofa::core::objectmodel;
    // These are the expected force residual if we do not activate any convergence threshold
    // and force the solve to do 10 Newton iterations
    const std::vector<double> expected_force_residual_norms = {
        2551.7326458060306,
        381.91182992522732,
        44.367042885694701,
        3.4122807603419312,
        0.40321094066950214,
        0.023070078859990586,
        0.0036006702672518268,
        0.00014283984897951575,
        0.000030818164546577138,
        0.0000012847345051708047
    };

    // Disable all convergence criteria
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);
    ;
    const std::vector<SReal> actual_force_residual_norms = this->execute().first;

    EXPECT_EQ(actual_force_residual_norms.size(), expected_force_residual_norms.size())
    << "The static ODE solver is supposed to execute 10 Newton steps since the convergence criteria were deactivated.";

    for (std::size_t newton_it = 0; newton_it < actual_force_residual_norms.size(); ++newton_it) {
        EXPECT_NEAR(actual_force_residual_norms[newton_it], expected_force_residual_norms[newton_it], 1e-3)
        << "The actual force residual norm doesn't match the expected one.";
    }
}

TEST_F(StaticSolverTest, RelativeResiduals) {
    using namespace sofa::core::objectmodel;
    // Disable all convergence criteria BUT the relative residual
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(1e-5);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);

    const sofa::type::vector<SReal> actual_force_residual_norms = this->execute().first;
    EXPECT_EQ(actual_force_residual_norms.size(), 6)
    << "The static ODE solver is supposed to converge after 6 Newton steps when using a relative residual threshold of 1e-5.\n"
    << actual_force_residual_norms;
}

TEST_F(StaticSolverTest, AbsoluteResiduals) {
    using namespace sofa::core::objectmodel;
    // Disable all convergence criteria BUT the absolute residual
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(1e-5);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);

    const sofa::type::vector<SReal> actual_force_residual_norms = this->execute().first;
    EXPECT_EQ(actual_force_residual_norms.size(), 10)
    << "The static ODE solver is supposed to converge after 10 Newton steps when using an absolute residual threshold of 1e-5.\n"
    << actual_force_residual_norms;
}

TEST_F(StaticSolverTest, Increments) {
    using namespace sofa::core::objectmodel;
    // These are the expected correction increment norms if we do not activate any convergence threshold
    // and force the solve to do 10 Newton iterations

    const std::vector<double> expected_increment_norms = {
        23.149116745347889,
        4.7185304859032309,
        0.95755125409873631,
        0.053244570231523215,
        0.0040133773093344888,
        0.0003421565281377296,
        0.000036312620114208186,
        0.0000019664565731698484,
        0.00000033108990100908061,
        0.0000000073088563878990673
    };

    // Disable all convergence criteria
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);
    ;
    const std::vector<SReal> actual_increment_norms = this->execute().second;

    EXPECT_EQ(actual_increment_norms.size(), expected_increment_norms.size())
    << "The static ODE solver is supposed to execute 10 Newton steps since the convergence criteria were deactivated.";

    for (std::size_t newton_it = 0; newton_it < actual_increment_norms.size(); ++newton_it) {
        EXPECT_NEAR(actual_increment_norms[newton_it], expected_increment_norms[newton_it], 1e-3)
        << "The actual increment norm doesn't match the expected one.";
    }
}

TEST_F(StaticSolverTest, RelativeIncrements) {
    using namespace sofa::core::objectmodel;
    // Disable all convergence criteria BUT the relative increment corrections
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(1e-5);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);

    const sofa::type::vector<SReal> actual_increment_norms = this->execute().second;
    EXPECT_EQ(actual_increment_norms.size(), 7)
    << "The static ODE solver is supposed to converge after 7 Newton steps when using a relative correction threshold of 1e-5.\n"
    << actual_increment_norms;
}

TEST_F(StaticSolverTest, AbsoluteIncrements) {
    using namespace sofa::core::objectmodel;
    // Disable all convergence criteria BUT the absolute increment corrections
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(1e-5);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);

    const sofa::type::vector<SReal> actual_increment_norms = this->execute().second;
    EXPECT_EQ(actual_increment_norms.size(), 8)
    << "The static ODE solver is supposed to converge after 8 Newton steps when using a relative correction threshold of 1e-5.\n"
    << actual_increment_norms;
}
