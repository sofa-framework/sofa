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

    void onSetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Direct");
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
        sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Integration");
        sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
        sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");

        sofa::simpleapi::createObject(m_scene.root, "DefaultAnimationLoop");
        sofa::simpleapi::createObject(m_scene.root, "BDF1");
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

        sofa::simpleapi::createObject(m_scene.root, "RestShapeSpringsForceField", {{"points", "0"}, {"spring", "0 1 " + std::to_string(k) + " 0 " + std::to_string(L_0) }});

        m_scene.initScene();
        EXPECT_EQ(m_solver->d_status.getValue(), component::odesolver::backward::NewtonStatus("Undefined"));

        NewtonRaphsonParameters params;
        params.maxNbIterationsNewton = 1;
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
