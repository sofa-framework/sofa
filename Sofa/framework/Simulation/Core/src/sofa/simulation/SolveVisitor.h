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
#include <sofa/core/MultiVecId.h>
#include <sofa/simulation/CpuTask.h>

#include <list>

namespace sofa::simulation
{

class SolveVisitorTask;

/** Used by the animation loop: send the solve signal to the others solvers
This visitor is able to run the solvers sequentially or concurrently.
 */
class SOFA_SIMULATION_CORE_API SolveVisitor : public Visitor
{
public:

    SolveVisitor(const sofa::core::ExecParams* params,
                 SReal _dt,
                 sofa::core::MultiVecCoordId X = sofa::core::VecCoordId::position(),
                 sofa::core::MultiVecDerivId V = sofa::core::VecDerivId::velocity(),
                 bool _parallelSolve = false,
                 bool computeForceIsolatedInteractionForceFields = false);

    SolveVisitor(const sofa::core::ExecParams* params, SReal _dt, bool free, bool _parallelSolve = false, bool computeForceIsolatedInteractionForceFields = false);

    virtual void processSolver(simulation::Node* node, sofa::core::behavior::OdeSolver* b);
    void fwdInteractionForceField(Node* node, core::behavior::BaseInteractionForceField* forceField);
    Result processNodeTopDown(simulation::Node* node) override;
    void processNodeBottomUp(simulation::Node* /*node*/) override;

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return true; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "behavior update position"; }
    const char* getClassName() const override { return "SolveVisitor"; }

    void setDt(SReal _dt);
    SReal getDt() const;

protected:
    SReal dt;
    sofa::core::MultiVecCoordId x;
    sofa::core::MultiVecDerivId v;
    bool m_parallelSolve {false };
    bool m_computeForceIsolatedInteractionForceFields { false };

    /// Container for the parallel tasks
    std::list<SolveVisitorTask> m_tasks;

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
class SolveVisitorTask : public sofa::simulation::CpuTask
{
public:
    SolveVisitorTask(sofa::simulation::CpuTask::Status* status,
                     sofa::core::behavior::OdeSolver* odeSolver,
                     const sofa::core::ExecParams* params,
                     SReal dt,
                     sofa::core::MultiVecCoordId x,
                     sofa::core::MultiVecDerivId v)
    : sofa::simulation::CpuTask(status)
    , m_solver(odeSolver)
    , m_execParams(params)
    , m_dt(dt)
    , m_x(x)
    , m_v(v)
    {}

    ~SolveVisitorTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

private:
    sofa::core::behavior::OdeSolver* m_solver {nullptr};
    const sofa::core::ExecParams* m_execParams {nullptr};
    SReal m_dt;
    sofa::core::MultiVecCoordId m_x;
    sofa::core::MultiVecDerivId m_v;
};

} // namespace sofa
