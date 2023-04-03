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

        if (! getSimulation()) {
            setSimulation(new DAGSimulation()) ;
        }

        root = getSimulation()->createNewNode("root");

        createObject(root, "RequiredPlugin", {{"pluginName", "Sofa.Component"}});
        createObject(root, "RegularGridTopology", {{"name", "grid"}, {"min", "-7.5 -7.5 0"}, {"max", "7.5 7.5 80"}, {"n", "3 3 9"}});
        auto s = createObject(root, "StaticSolver", {{"newton_iterations", "10"}});
        createObject(root, "SparseLDLSolver");
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
        createObject(root, "FixedConstraint", {{"indices", "@top_roi.indices"}});

        createObject(root, "BoxROI", {{"name", "base_roi"}, {"box", "-7.5 -7.5 79.9 7.5 7.5 80.1"}});
        createObject(root, "SurfacePressureForceField", {{"pressure", "100"}, {"mainDirection", "0 -1 0"}, {"triangleIndices", "@base_roi.trianglesInROI"}});

        solver = dynamic_cast<StaticSolver *> (s.get());
    }

    void onTearDown() override {
        getSimulation()->unload(root);
    }

    auto execute() -> std::pair<std::vector<SReal>, std::vector<SReal>> {
        using namespace std;
        getSimulation()->init(root.get());
        getSimulation()->animate(root.get(), 1);
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
    std::vector<double> expected_force_residual_norms = {
        1.237102e+03,
        6.931312e+00,
        2.634097e-01,
        2.829366e-02,
        2.928456e-03,
        3.017181e-04,
        3.108847e-05,
        3.203062e-06,
        3.300435e-07,
        3.398669e-08
    };

    // Disable all convergence criteria
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);
    ;
    std::vector<SReal> actual_force_residual_norms = this->execute().first;

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

    // The list of relative residuals are
    //       1             2             3             4             5             6             7             8             9             10
    // 1.000000e+00, 5.602864e-03, 2.129249e-04, 2.287093e-05, 2.367191e-06, 2.438911e-07, 2.513007e-08, 2.589194e-09,  2.668053e-10, 2.748057e-11
    // Setting a relative criterion of 1e-5 should therefore converge after the 5th Newton iteration.

    std::vector<SReal> actual_force_residual_norms = this->execute().first;
    EXPECT_EQ(actual_force_residual_norms.size(), 5)
    << "The static ODE solver is supposed to converge after 5 Newton steps when using a relative residual threshold of 1e-5.";
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

    // The list of relative residuals are
    //       1             2             3             4             5             6             7             8             9             10
    // 1.237102e+03, 6.931312e+00, 2.634097e-01, 2.829366e-02, 2.928456e-03, 3.017181e-04, 3.108845e-05, 3.203097e-06, 3.300652e-07, 3.399625e-08
    // Setting an absolute criterion of 1e-5 should therefore converge after the 8th Newton iteration.

    std::vector<SReal> actual_force_residual_norms = this->execute().first;
    EXPECT_EQ(actual_force_residual_norms.size(), 8)
    << "The static ODE solver is supposed to converge after 8 Newton steps when using an absolute residual threshold of 1e-5.";
}

TEST_F(StaticSolverTest, Increments) {
    using namespace sofa::core::objectmodel;
    // These are the expected correction increment norms if we do not activate any convergence threshold
    // and force the solve to do 10 Newton iterations
    std::vector<double> expected_increment_norms = {
            1.781729e+01,
            5.043276e-01,
            1.201313e-02,
            1.237255e-03,
            1.250073e-04,
            1.289042e-05,
            1.329191e-06,
            1.369818e-07,
            1.411500e-08,
            1.454314e-09,
    };

    // Disable all convergence criteria
    dynamic_cast< Data<unsigned> * > ( this->solver->findData("newton_iterations") )->setValue(10);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_correction_tolerance_threshold") )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("absolute_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<SReal> *   > ( this->solver->findData("relative_residual_tolerance_threshold")   )->setValue(-1);
    dynamic_cast< Data<bool> *     > ( this->solver->findData("should_diverge_when_residual_is_growing") )->setValue(false);
    ;
    std::vector<SReal> actual_increment_norms = this->execute().second;

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

    // The list of relative corrections are
    //       1             2             3             4             5             6             7             8             9             10
    // 1.000000e+00, 2.819276e-02, 6.711671e-04, 6.912001e-05, 6.983561e-06, 7.201258e-07, 7.425552e-08, 7.652517e-09,  7.885371e-10, 8.124548e-11
    // Setting a relative criterion of 1e-5 should therefore converge after the 5th Newton iteration.

    std::vector<SReal> actual_increment_norms = this->execute().second;
    EXPECT_EQ(actual_increment_norms.size(), 5)
    << "The static ODE solver is supposed to converge after 5 Newton steps when using a relative correction threshold of 1e-5.";
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

    // The list of absolute corrections are
    //       1             2             3             4             5             6             7             8             9             10
    // 1.781729e+01, 5.043276e-01, 1.201313e-02, 1.237255e-03, 1.250073e-04, 1.289042e-05, 1.329191e-06, 1.369819e-07, 1.411500e-08, 1.454313e-09
    // Setting an absolute criterion of 1e-5 should therefore converge after the 7th Newton iteration.

    std::vector<SReal> actual_increment_norms = this->execute().second;
    EXPECT_EQ(actual_increment_norms.size(), 7)
    << "The static ODE solver is supposed to converge after 7 Newton steps when using a relative correction threshold of 1e-5.";
}
