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
#pragma once

#include <sofa/simulation/Visitor.h>
#include <sofa/core/behavior/BaseIntegrationScheme.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/simulation/task/CpuTask.h>

#include <list>

namespace sofa::simulation
{

class IntegrateVisitorTask;

/** Used by the animation loop: send the solve signal to the others solvers
This visitor is able to run the solvers sequentially or concurrently.
 */
class SOFA_SIMULATION_CORE_API IntegrateVisitor : public Visitor
{
public:

    enum TaskType
    {
        SETUP_INTEGRATION_STEP,
        COMPUTE_LHS,
        COMPUTE_RHS,
        EVALUATE_RESIDUAL,
        SOLVE_LINEAR_EQUATION,
        UPDATE_STATE_FROM_LINEAR_SOLUTION,
        FINALIZE_INTEGRATION_STEP,
    };


    IntegrateVisitor(const sofa::core::ExecParams* params,
                 SReal _dt,
                 TaskType taskType,
                 bool firstStep,
                 SReal alpha,
                 sofa::core::MultiVecCoordId X = sofa::core::vec_id::write_access::position,
                 sofa::core::MultiVecDerivId V = sofa::core::vec_id::write_access::velocity,
                 bool _parallelSolve = false,
                 bool computeForceIsolatedInteractionForceFields = false);

    IntegrateVisitor(const sofa::core::ExecParams* params, SReal _dt, bool free, bool _parallelSolve = false, bool computeForceIsolatedInteractionForceFields = false);


    virtual void processSolver(simulation::Node* node,  sofa::core::behavior::BaseIntegrationScheme* b);
    void fwdInteractionForceField(Node* node, core::behavior::BaseInteractionForceField* forceField);

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "behavior update position"; }
    const char* getClassName() const override { return "IntegrateVisitor"; }


    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* /*node*/) override;


protected:
    SReal m_dt;
    TaskType m_taskType;
    bool m_firstStep;
    SReal m_alpha;
    sofa::core::MultiVecCoordId m_xId;
    sofa::core::MultiVecDerivId m_vId;


    bool m_parallelSolve {false };
    bool m_computeForceIsolatedInteractionForceFields { false };

    /// Container for the parallel tasks
    std::list<IntegrateVisitorTask> m_tasks;

    /// Status for the parallel tasks
    sofa::simulation::CpuTask::Status m_status;

    /// Function called if the solvers run sequentially
    void sequentialSolve(simulation::Node* node);

    /// Function called if the solvers run concurrently
    /// Solving tasks are added to the list of tasks and start to run.
    /// However, there is no check that the tasks finished. This is
    /// done later, once all nodes have been traversed.
    void parallelSolve(simulation::Node* node);

    /// Initialize the task scheduler if it is not done already
    void initializeTaskScheduler();
};

/// A task to provide to a task scheduler in which a solver solves
class IntegrateVisitorTask : public sofa::simulation::CpuTask
{
public:
    IntegrateVisitorTask(sofa::simulation::CpuTask::Status* status,
                      sofa::core::behavior::BaseIntegrationScheme* IntegrationScheme,
                     const sofa::core::ExecParams* params,
                     SReal dt,
                     sofa::core::MultiVecCoordId x,
                     sofa::core::MultiVecDerivId v)
    : sofa::simulation::CpuTask(status)
    , m_solver(IntegrationScheme)
    , m_execParams(params)
    , m_dt(dt)
    , m_x(x)
    , m_v(v)
    {}

    ~IntegrateVisitorTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

private:
     sofa::core::behavior::BaseIntegrationScheme* m_solver {nullptr};
    const sofa::core::ExecParams* m_execParams {nullptr};
    SReal m_dt;
    sofa::core::MultiVecCoordId m_x;
    sofa::core::MultiVecDerivId m_v;
};

class IntegrationHelper
{
public:

    IntegrateVisitor(const sofa::core::ExecParams* params,
                 SReal _dt,
                 sofa::core::MultiVecCoordId X = sofa::core::vec_id::write_access::position,
                 sofa::core::MultiVecDerivId V = sofa::core::vec_id::write_access::velocity,
                 bool _parallelSolve = false,
                 bool computeForceIsolatedInteractionForceFields = false);

    void setupIntegrationStep();
    SReal evaluateResidual();
    void computeLHS(bool firstIteration);
    void computeRHS(bool firstIteration);
    void solveLinearEquation();
    void updateStatesFromLinearSolution(SReal alpha, bool firstIteration = false);


    SReal m_dt;
    core::MultiVecCoordId m_xId;
    core::MultiVecDerivId m_vId;
    bool m_parallelSolve;
    bool m_computeForceIsolatedInteractionForceFields;
};

} // namespace sofa
