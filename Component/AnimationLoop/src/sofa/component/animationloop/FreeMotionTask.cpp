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

#include <sofa/component/animationloop/FreeMotionTask.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/SolveVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/simulation/MechanicalOperations.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalVOpVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalVOpVisitor;

namespace sofa::component::animationloop 
{

FreeMotionTask::FreeMotionTask(sofa::simulation::Node* node,
                               const sofa::core::ExecParams* params,
                               const core::ConstraintParams* cparams,
                               SReal dt,
                               sofa::core::MultiVecId pos,
                               sofa::core::MultiVecId freePos,
                               sofa::core::MultiVecDerivId freeVel,
                               simulation::common::MechanicalOperations* mop,
                               sofa::core::objectmodel::BaseContext* context,
                               sofa::simulation::CpuTask::Status* status,
                               bool parallelSolve)
    : sofa::simulation::CpuTask(status)
    , m_node(node)
    , m_params(params)
    , m_cparams(cparams)
    , m_dt(dt)
    , m_pos(pos)
    , m_freePos(freePos)
    , m_freeVel(freeVel)
    , m_mop(mop)
    , m_context(context)
    , m_parallelSolve(parallelSolve)
{}

sofa::simulation::Task::MemoryAlloc FreeMotionTask::run()
{
    {
        sofa::helper::ScopedAdvancedTimer timer("FreeMotion");
        simulation::SolveVisitor freeMotion(m_params, m_dt, true, m_parallelSolve);
        m_node->execute(&freeMotion);
    }

    m_mop->projectResponse(m_freeVel);
    m_mop->propagateDx(m_freeVel, true);

    if (m_cparams->constOrder() == sofa::core::ConstraintParams::POS ||
        m_cparams->constOrder() == sofa::core::ConstraintParams::POS_AND_VEL)
    {
        sofa::helper::ScopedAdvancedTimer timer("freePosEqPosPlusFreeVelDt");
        MechanicalVOpVisitor freePosEqPosPlusFreeVelDt(m_params, m_freePos, m_pos, m_freeVel, m_dt);
        freePosEqPosPlusFreeVelDt.setMapped(true);
        m_context->executeVisitor(&freePosEqPosPlusFreeVelDt);
    }

    return simulation::Task::Stack;
}

} // namespace sofa::component::animationloop
