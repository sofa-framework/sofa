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

#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/BaseIntegrationScheme.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/simulation/integrationscheme/ImplicitIntegrationScheme.h>
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
        NONE,
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
                 bool _parallelExecution = false,
                 bool computeForceIsolatedInteractionForceFields = false);


    void fwdInteractionForceField(Node* node, core::behavior::BaseInteractionForceField* forceField);

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "behavior update position"; }
    const char* getClassName() const override { return "IntegrateVisitor"; }


    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* /*node*/) override;

    void parallelWaitForResult();

    SReal getResult() const
    {
        return m_result;
    }

    void setFirstIteration(bool firstStep)
    {
        m_firstStep = firstStep;
    }

    void setAlpha(SReal alpha)
    {
        m_alpha = alpha;
    }

    void setDt(SReal dt)
    {
        m_dt = dt;
    }

    void setTaskType(TaskType taskType)
    {
        m_taskType = taskType;
    }

    bool isExecutionParallel() const
    {
        return m_parallelExecution;
    }


    static SReal s_doProcessIS(sofa::core::behavior::BaseIntegrationScheme* integrationScheme,
                                 const sofa::core::ExecParams* params,
                                 const sofa::simulation::Node* node,
                                 SReal dt,
                                 TaskType taskType,
                                 bool firstStep = true,
                                 SReal alpha = 1.0,
                                 sofa::core::MultiVecCoordId X = sofa::core::vec_id::write_access::position,
                                 sofa::core::MultiVecDerivId V = sofa::core::vec_id::write_access::velocity);


protected:
    SReal m_dt;
    TaskType m_taskType;
    bool m_firstStep;
    SReal m_alpha;
    sofa::core::MultiVecCoordId m_xId;
    sofa::core::MultiVecDerivId m_vId;

    SReal m_result;

    bool m_parallelExecution {false };
    bool m_computeForceIsolatedInteractionForceFields { false };

    std::list<IntegrateVisitorTask> m_tasks;
    sofa::simulation::CpuTask::Status m_status;

    void sequentialExecute(simulation::Node* node);
    void parallelExecute(simulation::Node* node);

    void initializeTaskScheduler();
};

/// A task to provide to a task scheduler in which a solver solves
class IntegrateVisitorTask : public sofa::simulation::CpuTask
{
public:
    IntegrateVisitorTask(sofa::simulation::CpuTask::Status* status,
                         std::function<SReal()> task)
    : sofa::simulation::CpuTask(status)
    , m_task(task)
    {}

    ~IntegrateVisitorTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final
    {
        m_result = m_task();
        return Task::Stack;
    }

    SReal getResult() const
    {
        return m_result;
    }


private:
    std::function<SReal()> m_task;
    SReal m_result;
};

class IntegrationHelper
{
public:

    IntegrationHelper(const sofa::core::ExecParams* params,
                 sofa::core::MultiVecCoordId X = sofa::core::vec_id::write_access::position,
                 sofa::core::MultiVecDerivId V = sofa::core::vec_id::write_access::velocity,
                 bool parallelSolve = false,
                 bool computeForceIsolatedInteractionForceFields = false);

    void setupIntegrationStep(sofa::simulation::Node * node);
    SReal evaluateResidual(sofa::simulation::Node * node);
    void computeLHS(sofa::simulation::Node * node, bool firstIteration);
    void computeRHS(sofa::simulation::Node * node, bool firstIteration);
    void solveLinearEquation(sofa::simulation::Node * node);
    void updateStatesFromLinearSolution(sofa::simulation::Node * node, SReal alpha, bool firstIteration = false);


private:
    IntegrateVisitor m_visitor;
};

} // namespace sofa
