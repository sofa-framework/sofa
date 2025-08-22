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
#include <sofa/component/odesolver/backward/NewtonRaphsonSolver.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/testing/BaseSimulationTest.h>
#include <sofa/testing/NumericTest.h>

namespace sofa
{

struct NewtonRaphsonSquareRootTest : public testing::NumericTest<SReal>
{
    void doSetUp() override
    {
        root = simulation::getSimulation()->createNewNode("root");

        solver = core::objectmodel::New <component::odesolver::backward::NewtonRaphsonSolver>();
        root->addObject(solver);

        solver->d_relativeInitialStoppingThreshold.setValue(-1);
        solver->d_absoluteResidualStoppingThreshold.setValue(-1);
        solver->d_relativeSuccessiveStoppingThreshold.setValue(-1);
        solver->d_maxNbIterationsLineSearch.setValue(1);
        solver->d_maxNbIterationsLineSearch.setValue(false);
        solver->f_printLog.setValue(false);

        simulation::node::initRoot(root.get());
    }

    sofa::type::vector<SReal> computeSquareRoot(const SReal x, const SReal initialGuess) const
    {
        struct SquareRootFunction : public component::odesolver::backward::newton_raphson::BaseNonLinearFunction
        {
            void setInitialGuess(SReal initialGuess)
            {
                m_currentGuess = initialGuess;
                m_guesses.push_back(initialGuess);
            }

            void evaluateCurrentGuess() override
            {
                const SReal x2 = m_currentGuess * m_currentGuess;
                m_currentEvaluation = x2 - m_positiveNumber;
            }

            SReal squaredNormLastEvaluation() override
            {
                return m_currentEvaluation * m_currentEvaluation;
            }

            void computeGradientFromCurrentGuess() override
            {
                m_gradient = 2 * m_currentGuess;
            }

            void updateGuessFromLinearSolution(SReal alpha) override
            {
                m_currentGuess += alpha * m_linearSolverSolution;
            }

            void solveLinearEquation() override
            {
                m_linearSolverSolution = -m_currentEvaluation / m_gradient;
            }

            SReal squaredNormDx() override
            {
                return m_linearSolverSolution * m_linearSolverSolution;
            }

            SReal squaredLastEvaluation() override
            {
                return m_currentGuess * m_currentGuess;
            }

            void endNewtonIteration() override
            {
                m_guesses.push_back(m_currentGuess);
            }

            sofa::type::vector<SReal> m_guesses;
            SReal m_currentGuess = 0;
            SReal m_currentEvaluation = 0;
            SReal m_gradient = 0;
            SReal m_linearSolverSolution = 0;
            SReal m_positiveNumber;

        } squareRootFunction;

        squareRootFunction.m_positiveNumber = x;
        squareRootFunction.setInitialGuess(initialGuess);

        solver->solve(squareRootFunction);

        return squareRootFunction.m_guesses;
    }

    simulation::Node::SPtr root ;
    component::odesolver::backward::NewtonRaphsonSolver::SPtr solver;
};

TEST_F(NewtonRaphsonSquareRootTest, squareRoot612_1)
{
    solver->d_maxNbIterationsNewton.setValue(9);

    const auto sqrt = this->computeSquareRoot(612, 1);

    static const sofa::type::vector<SReal> sequenceGuesses {
        1,
        306.5,
        154.2483686786,
        79.1079978644,
        43.4221286822,
        28.7581624288,
        25.0195385369,
        24.7402106712,
        24.7386338040,
        24.7386337537
    };

    ASSERT_EQ(sequenceGuesses.size(), sqrt.size());

    for (unsigned int i = 0; i < sequenceGuesses.size(); ++i)
    {
        EXPECT_NEAR(sequenceGuesses[i], sqrt[i], 1e-10);
    }
}

TEST_F(NewtonRaphsonSquareRootTest, squareRoot612_10)
{
    solver->d_maxNbIterationsNewton.setValue(5);

    const auto sqrt = this->computeSquareRoot(612, 10);

    static const sofa::type::vector<SReal> sequenceGuesses {
        10,
        35.6,
        26.3955056180,
        24.7906354925,
        24.7386882941,
        24.7386337538
    };

    ASSERT_EQ(sequenceGuesses.size(), sqrt.size());

    for (unsigned int i = 0; i < sequenceGuesses.size(); ++i)
    {
        EXPECT_NEAR(sequenceGuesses[i], sqrt[i], 1e-10);
    }
}

TEST_F(NewtonRaphsonSquareRootTest, squareRoot612_minus20)
{
    solver->d_maxNbIterationsNewton.setValue(4);

    const auto sqrt = this->computeSquareRoot(612, -20);

    static const sofa::type::vector<SReal> sequenceGuesses {
        -20,
        -25.3,
        -24.7448616601,
        -24.7386345374,
        -24.7386337537
    };

    ASSERT_EQ(sequenceGuesses.size(), sqrt.size());

    for (unsigned int i = 0; i < sequenceGuesses.size(); ++i)
    {
        EXPECT_NEAR(sequenceGuesses[i], sqrt[i], 1e-10);
    }
}





struct NewtonRaphsonParameters
{
    SReal relativeSuccessiveStoppingThreshold {};
    SReal relativeInitialStoppingThreshold {};
    SReal absoluteResidualStoppingThreshold {};
    SReal relativeEstimateDifferenceThreshold {};
    SReal absoluteEstimateDifferenceThreshold {};
    unsigned int maxNbIterationsLineSearch = 1;
    unsigned int maxNbIterationsNewton = 1;

    void apply(component::odesolver::backward::NewtonRaphsonSolver* solver) const
    {
        solver->d_relativeSuccessiveStoppingThreshold.setValue(relativeSuccessiveStoppingThreshold);
        solver->d_relativeInitialStoppingThreshold.setValue(relativeInitialStoppingThreshold);
        solver->d_absoluteResidualStoppingThreshold.setValue(absoluteResidualStoppingThreshold);
        solver->d_relativeEstimateDifferenceThreshold.setValue(relativeEstimateDifferenceThreshold);
        solver->d_absoluteEstimateDifferenceThreshold.setValue(absoluteEstimateDifferenceThreshold);
        solver->d_maxNbIterationsLineSearch.setValue(maxNbIterationsLineSearch);
        solver->d_maxNbIterationsNewton.setValue(maxNbIterationsNewton);
    }
};

struct NewtonRaphsonTest : public testing::BaseSimulationTest
{
    SceneInstance m_scene{};

    component::odesolver::backward::NewtonRaphsonSolver* m_solver { nullptr };
    component::statecontainer::MechanicalObject<defaulttype::Vec1Types>* m_state { nullptr };

    void doSetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Direct");
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");

        sofa::simpleapi::createObject(m_scene.root, "DefaultAnimationLoop");
        sofa::simpleapi::createObject(m_scene.root, "BDFOdeSolver", {{"order", "1"}});
        auto solverObject = sofa::simpleapi::createObject(m_scene.root, "NewtonRaphsonSolver", {{"printLog", "true"}});
        m_solver = dynamic_cast<component::odesolver::backward::NewtonRaphsonSolver*>(solverObject.get());
        sofa::simpleapi::createObject(m_scene.root, "EigenSimplicialLDLT", {{"template", "CompressedRowSparseMatrix"}});
        auto stateObject = sofa::simpleapi::createObject(m_scene.root, "MechanicalObject", {{"template", "Vec1"}, {"position", "1"}, {"rest_position", "0"}});
        m_state = dynamic_cast<component::statecontainer::MechanicalObject<defaulttype::Vec1Types>*>(stateObject.get());
        sofa::simpleapi::createObject(m_scene.root, "PointSetTopologyContainer", {{"name", "topology"}, {"position", "1"}});

        ASSERT_NE(nullptr, m_solver);
        ASSERT_NE(nullptr, m_state);
    }

    void noForce()
    {
        m_scene.initScene();
        EXPECT_EQ(m_solver->d_status.getValue(), component::odesolver::backward::NewtonStatus("Undefined"));

        m_scene.simulate(0.01_sreal);
        EXPECT_EQ(m_solver->d_status.getValue(), component::odesolver::backward::NewtonStatus("ConvergedEquilibrium"));
    }

    void gravity(const NewtonRaphsonParameters& params, const component::odesolver::backward::NewtonStatus& expectedStatus)
    {
        static constexpr SReal gravity = -1;
        m_scene.root->setGravity({gravity, 0, 0});

        sofa::simpleapi::importPlugin("Sofa.Component.Mass");
        sofa::simpleapi::createObject(m_scene.root, "UniformMass", {{"totalMass", "1"}, {"topology", "@topology"}, {"printLog", "true"}});
        params.apply(m_solver);

        m_scene.initScene();
        EXPECT_EQ(m_solver->d_status.getValue(), component::odesolver::backward::NewtonStatus("Undefined"));

        auto previousVelocity = m_state->read(sofa::core::vec_id::read_access::velocity)->getValue();
        auto previousPosition = m_state->read(sofa::core::vec_id::read_access::position)->getValue();

        static constexpr SReal dt = 0.1_sreal;
        m_scene.simulate(dt);
        EXPECT_EQ(m_solver->d_status.getValue(), expectedStatus);

        const auto& velocity = m_state->read(sofa::core::vec_id::read_access::velocity)->getValue();
        const auto& position = m_state->read(sofa::core::vec_id::read_access::position)->getValue();

        EXPECT_FLOATINGPOINT_EQ(velocity[0].x(), gravity * dt)
        EXPECT_FLOATINGPOINT_EQ(position[0].x(), previousPosition[0].x() + velocity[0].x() * dt);

        for (unsigned int i = 1; i < 10; ++i)
        {
            previousVelocity = m_state->read(sofa::core::vec_id::read_access::velocity)->getValue();
            previousPosition = m_state->read(sofa::core::vec_id::read_access::position)->getValue();

            m_scene.simulate(dt);
            EXPECT_FLOATINGPOINT_EQ(velocity[0].x(), previousVelocity[0].x() + gravity * dt);

            EXPECT_FLOATINGPOINT_EQ(position[0].x(), previousPosition[0].x() + velocity[0].x() * dt);
        }
    }

    void spring(const SReal k, const SReal L_0, const SReal dt)
    {
        sofa::simpleapi::importPlugin("Sofa.Component.SolidMechanics.Spring");

        sofa::simpleapi::createObject(m_scene.root, "RestShapeSpringsForceField", {{"points", "0"}, {"stiffness", std::to_string(k)}});

        m_scene.initScene();
        EXPECT_EQ(m_solver->d_status.getValue(), component::odesolver::backward::NewtonStatus("Undefined"));

        NewtonRaphsonParameters params;
        params.maxNbIterationsNewton = 2;
        params.apply(m_solver);
        
        m_scene.simulate(dt);

        EXPECT_EQ(m_solver->d_status.getValue(), component::odesolver::backward::NewtonStatus("DivergedMaxIterations"));

    }

};

TEST_F(NewtonRaphsonTest, noForce)
{
    this->noForce();
}

TEST_F(NewtonRaphsonTest, gravity_noStopping)
{
    // no stopping criteria is defined so it cannot converge
    this->gravity(
        NewtonRaphsonParameters{},
        component::odesolver::backward::NewtonStatus("DivergedMaxIterations"));
}

TEST_F(NewtonRaphsonTest, gravity_relativeSuccessiveStopping)
{
    NewtonRaphsonParameters params;
    params.relativeSuccessiveStoppingThreshold = 1e-15_sreal;

    this->gravity(
        params,
        component::odesolver::backward::NewtonStatus("ConvergedResidualSuccessiveRatio"));
}

TEST_F(NewtonRaphsonTest, gravity_relativeInitialStopping)
{
    NewtonRaphsonParameters params;
    params.relativeInitialStoppingThreshold = 1e-15_sreal;

    this->gravity(
        params,
        component::odesolver::backward::NewtonStatus("ConvergedResidualInitialRatio"));
}

TEST_F(NewtonRaphsonTest, gravity_absoluteResidualStopping)
{
    NewtonRaphsonParameters params;
    params.absoluteResidualStoppingThreshold = 1e-15_sreal;

    this->gravity(
        params,
        component::odesolver::backward::NewtonStatus("ConvergedAbsoluteResidual"));
}

TEST_F(NewtonRaphsonTest, gravity_relativeEstimateDifferenceStopping)
{
    NewtonRaphsonParameters params;
    params.relativeEstimateDifferenceThreshold = 1e-15_sreal;

    // considered as diverged because this threshold requires more than one iteration
    this->gravity(
        params,
        component::odesolver::backward::NewtonStatus("DivergedMaxIterations"));
}

TEST_F(NewtonRaphsonTest, gravity_absoluteEstimateDifferenceStopping)
{
    NewtonRaphsonParameters params;
    params.absoluteEstimateDifferenceThreshold = 1e-15_sreal;

    // considered as diverged because this threshold requires more than one iteration
    this->gravity(
        params,
        component::odesolver::backward::NewtonStatus("DivergedMaxIterations"));
}

TEST_F(NewtonRaphsonTest, spring)
{
    this->spring(1000, 1, 0.1);
}


}
