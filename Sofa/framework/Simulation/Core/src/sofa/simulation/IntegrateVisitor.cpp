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
#include <sofa/simulation/IntegrateVisitor.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>

namespace sofa::simulation
{

SReal IntegrateVisitor::s_doProcessIS(sofa::core::behavior::BaseIntegrationScheme* integrationScheme,
                                 const sofa::core::ExecParams* params,
                                 const sofa::simulation::Node* node,
                                 SReal dt,
                                 TaskType taskType,
                                 bool firstStep,
                                 SReal alpha,
                                 sofa::core::MultiVecCoordId X,
                                 sofa::core::MultiVecDerivId V)
{
    helper::ScopedAdvancedTimer timer("Mechanical",node);

    if (!integrationScheme->isImplicit())
        return 0;

    auto implicitIntegrationScheme = static_cast<sofa::simulation::integrationscheme::ImplicitIntegrationScheme*>(integrationScheme);


    switch (taskType)
    {
        case SETUP_INTEGRATION_STEP:
        {
            implicitIntegrationScheme->setupIntegrationStep(params, dt, X, V);
            return 0;
        }
        case COMPUTE_LHS:
        {
            implicitIntegrationScheme->computeLHS(firstStep);
            return 0;
        }
        case COMPUTE_RHS:
        {
            implicitIntegrationScheme->computeRHS(firstStep);
            return 0;
        }
        case EVALUATE_RESIDUAL:
        {
            const SReal res = implicitIntegrationScheme->evaluateResidual();
            return res;
        }
        case SOLVE_LINEAR_EQUATION:
        {
            implicitIntegrationScheme->solveLinearEquation();
            return 0;
        }
        case UPDATE_STATE_FROM_LINEAR_SOLUTION:
        {
            implicitIntegrationScheme->updateStatesFromLinearSolution(alpha,firstStep);
            return 0;
        }
        case FINALIZE_INTEGRATION_STEP:
        {
            implicitIntegrationScheme->finalizeIntegrationStep();
            return 0;
        }
        default:
            return 0;
    }
};

void IntegrateVisitor::fwdInteractionForceField(Node* node, core::behavior::BaseInteractionForceField* forceField)
{
    SOFA_UNUSED(node);

    const core::MultiVecDerivId ffId = core::vec_id::write_access::externalForce;
    core::MechanicalParams mparams;
    mparams.setDt(m_dt);
    forceField->addForce(&mparams, ffId);
}

Visitor::Result IntegrateVisitor::processNodeTopDown(simulation::Node* node)
{
    if (! node->integrationScheme.empty())
    {
        if (m_parallelExecution)
        {
            parallelExecute(node);
        }
        else
        {
            sequentialExecute(node);
        }
        return RESULT_PRUNE;
    }

    if (m_computeForceIsolatedInteractionForceFields and m_taskType == COMPUTE_RHS)
    {
        for_each(this, node, node->interactionForceField, &IntegrateVisitor::fwdInteractionForceField);
    }
    return RESULT_CONTINUE;
}

void IntegrateVisitor::processNodeBottomUp(simulation::Node*)
{
    // only in case of parallel solving:
    // processNodeBottomUp is called after all processNodeTopDown calls are done,
    // i.e when all parallel tasks have been created and started.
    // It is time to wait them to finish

    if (!m_tasks.empty())
    {
        auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
        assert(taskScheduler != nullptr);
        SCOPED_TIMER_VARNAME(parallelSolveTimer, "waitParallelTasks");
        taskScheduler->workUntilDone(&m_status);
    }
    m_tasks.clear();
}

IntegrateVisitor::IntegrateVisitor(const sofa::core::ExecParams* params, SReal _dt, TaskType taskType,
                 bool firstStep, SReal alpha, sofa::core::MultiVecCoordId X, sofa::core::MultiVecDerivId V,
                 bool _parallelExecution, bool computeForceIsolatedInteractionForceFields)

        : Visitor(params)
        , m_dt(_dt)
        , m_taskType(taskType)
        , m_firstStep(firstStep)
        , m_alpha(alpha)
        , m_xId(X)
        , m_vId(V)
        , m_parallelExecution(_parallelExecution)
        , m_computeForceIsolatedInteractionForceFields(computeForceIsolatedInteractionForceFields)
{
    if (m_parallelExecution)
    {
        initializeTaskScheduler();
    }
}


void IntegrateVisitor::sequentialExecute(simulation::Node* node)
{
    for (auto* solver : node->integrationScheme)
    {
        m_result += s_doProcessIS(solver, this->params, node, this->m_dt, this->m_taskType, this->m_firstStep, this->m_alpha, this->m_xId, this->m_vId );
    }
}

void IntegrateVisitor::parallelExecute(simulation::Node* node)
{
    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);

    for (auto* solver : node->integrationScheme)
    {
        m_tasks.emplace_back(&m_status, [this, solver, node]()->SReal {return s_doProcessIS(solver, this->params, node, this->m_dt, this->m_taskType, this->m_firstStep, this->m_alpha, this->m_xId, this->m_vId );});
        taskScheduler->addTask(&m_tasks.back());
    }
}

void IntegrateVisitor::parallelWaitForResult()
{
    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);
    taskScheduler->workUntilDone(&m_status);
    for (auto& task : m_tasks)
    {
        m_result += task.getResult();
    }
    m_tasks.clear();
}

void IntegrateVisitor::initializeTaskScheduler()
{
    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);
    if (taskScheduler->getThreadCount() < 1)
    {
        taskScheduler->init(0);
    }
}


IntegrationHelper::IntegrationHelper(const sofa::core::ExecParams* params,
                 sofa::core::MultiVecCoordId X ,
                 sofa::core::MultiVecDerivId V ,
                 bool parallelExecution ,
                 bool computeForceIsolatedInteractionForceFields )
    : m_visitor(params, 0.0, IntegrateVisitor::TaskType::NONE, true, 1.0, X, V, parallelExecution, computeForceIsolatedInteractionForceFields)
{}

void  IntegrationHelper::setupIntegrationStep(sofa::simulation::Node * node)
{
    helper::ScopedAdvancedTimer timer("SetupIntegrationStep");
    m_visitor.setDt(node->getDt());
    m_visitor.setTaskType(IntegrateVisitor::TaskType::SETUP_INTEGRATION_STEP);
    m_visitor.execute(node);
    if (m_visitor.isExecutionParallel())
        m_visitor.parallelWaitForResult();
}

SReal IntegrationHelper::evaluateResidual(sofa::simulation::Node * node)
{
    helper::ScopedAdvancedTimer timer("evaluateResidual");
    m_visitor.setDt(node->getDt());
    m_visitor.setTaskType(IntegrateVisitor::TaskType::EVALUATE_RESIDUAL);
    m_visitor.execute(node);
    if (m_visitor.isExecutionParallel())
        m_visitor.parallelWaitForResult();
    return m_visitor.getResult();
}

void  IntegrationHelper::computeLHS(sofa::simulation::Node * node, bool firstIteration)
{
    helper::ScopedAdvancedTimer timer("computeLHS");
    m_visitor.setDt(node->getDt());
    m_visitor.setTaskType(IntegrateVisitor::TaskType::COMPUTE_LHS);
    m_visitor.setFirstIteration(firstIteration);
    m_visitor.execute(node);
    if (m_visitor.isExecutionParallel())
        m_visitor.parallelWaitForResult();
}

void  IntegrationHelper::computeRHS(sofa::simulation::Node * node, bool firstIteration)
{
    helper::ScopedAdvancedTimer timer("computeRHS");
    m_visitor.setDt(node->getDt());
    m_visitor.setTaskType(IntegrateVisitor::TaskType::COMPUTE_RHS);
    m_visitor.setFirstIteration(firstIteration);
    m_visitor.execute(node);
    if (m_visitor.isExecutionParallel())
        m_visitor.parallelWaitForResult();
}

void  IntegrationHelper::solveLinearEquation(sofa::simulation::Node * node)
{
    helper::ScopedAdvancedTimer timer("solveLinearEquation");
    m_visitor.setDt(node->getDt());
    m_visitor.setTaskType(IntegrateVisitor::TaskType::SOLVE_LINEAR_EQUATION);
    m_visitor.execute(node);
    if (m_visitor.isExecutionParallel())
        m_visitor.parallelWaitForResult();
}

void  IntegrationHelper::updateStatesFromLinearSolution(sofa::simulation::Node * node, SReal alpha, bool firstIteration)
{
    helper::ScopedAdvancedTimer timer("updateStatesFromLinearSolution");
    m_visitor.setDt(node->getDt());
    m_visitor.setTaskType(IntegrateVisitor::TaskType::UPDATE_STATE_FROM_LINEAR_SOLUTION);
    m_visitor.setFirstIteration(firstIteration);
    m_visitor.setAlpha(alpha);
    m_visitor.execute(node);
    if (m_visitor.isExecutionParallel())
        m_visitor.parallelWaitForResult();
}



} // namespace sofa::simulation

