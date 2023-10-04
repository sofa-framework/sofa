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
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>

namespace sofa::simulation
{

void SolveVisitor::processSolver(simulation::Node* node, sofa::core::behavior::OdeSolver* s)
{
    helper::ScopedAdvancedTimer timer("Mechanical",node);
    s->solve(params, dt, x, v);
}

void SolveVisitor::fwdInteractionForceField(Node* node, core::behavior::BaseInteractionForceField* forceField)
{
    SOFA_UNUSED(node);

    const core::MultiVecDerivId ffId = core::VecDerivId::externalForce();
    core::MechanicalParams mparams;
    mparams.setDt(dt);
    forceField->addForce(&mparams, ffId);
}

Visitor::Result SolveVisitor::processNodeTopDown(simulation::Node* node)
{
    if (! node->solver.empty())
    {
        if (m_parallelSolve)
        {
            parallelSolve(node);
        }
        else
        {
            sequentialSolve(node);
        }
        return RESULT_PRUNE;
    }

    if (m_computeForceIsolatedInteractionForceFields)
    {
        for_each(this, node, node->interactionForceField, &SolveVisitor::fwdInteractionForceField);
    }
    return RESULT_CONTINUE;
}

void SolveVisitor::processNodeBottomUp(simulation::Node*)
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

void SolveVisitor::setDt(SReal _dt)
{
    dt = _dt;
}

SReal SolveVisitor::getDt() const
{
    return dt;
}

SolveVisitor::SolveVisitor(const sofa::core::ExecParams* params, SReal _dt, sofa::core::MultiVecCoordId X,
                           sofa::core::MultiVecDerivId V, bool _parallelSolve, bool computeForceIsolatedInteractionForceFields)

        : Visitor(params)
        , dt(_dt)
        , x(X)
        , v(V)
        , m_parallelSolve(_parallelSolve)
        , m_computeForceIsolatedInteractionForceFields(computeForceIsolatedInteractionForceFields)
{
    if (m_parallelSolve)
    {
        initializeTaskScheduler();
    }
}

SolveVisitor::SolveVisitor(const sofa::core::ExecParams* params, SReal _dt, bool free, bool _parallelSolve, bool computeForceIsolatedInteractionForceFields)
: Visitor(params), dt(_dt), m_parallelSolve(_parallelSolve), m_computeForceIsolatedInteractionForceFields(computeForceIsolatedInteractionForceFields)
{
    if(free)
    {
        x = sofa::core::VecCoordId::freePosition();
        v = sofa::core::VecDerivId::freeVelocity();
    }
    else
    {
        x = sofa::core::VecCoordId::position();
        v = sofa::core::VecDerivId::velocity();
    }

    if (m_parallelSolve)
    {
        initializeTaskScheduler();
    }
}

void SolveVisitor::sequentialSolve(simulation::Node* node)
{
    for_each(this, node, node->solver, &SolveVisitor::processSolver);
}

void SolveVisitor::parallelSolve(simulation::Node* node)
{
    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);

    for (auto* solver : node->solver)
    {
        m_tasks.emplace_back(&m_status, solver, params, dt, x, v);
        taskScheduler->addTask(&m_tasks.back());
    }
}

void SolveVisitor::initializeTaskScheduler()
{
    auto* taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(taskScheduler != nullptr);
    if (taskScheduler->getThreadCount() < 1)
    {
        taskScheduler->init(0);
    }
}

sofa::simulation::Task::MemoryAlloc SolveVisitorTask::run()
{
    m_solver->solve(m_execParams, m_dt, m_x, m_v);
    return Task::Stack;
}

} // namespace sofa::simulation

