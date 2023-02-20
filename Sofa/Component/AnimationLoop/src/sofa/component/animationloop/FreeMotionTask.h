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

#include <sofa/component/animationloop/config.h>

#ifndef SOFA_BUILD_SOFA_COMPONENT_ANIMATIONLOOP
SOFA_DEPRECATED_HEADER_NOT_REPLACED("v22.12", "v23.06")
#endif

#include <sofa/simulation/CpuTask.h>

#include <sofa/simulation/fwd.h>
#include <sofa/core/fwd.h>
#include <sofa/core/MultiVecId.h>

namespace sofa::component::animationloop
{

class SOFA_ATTRIBUTE_DEPRECATED_FREEMOTIONTASK() FreeMotionTask : public sofa::simulation::CpuTask
{
public:
    FreeMotionTask(
            sofa::simulation::Node* node,
            const sofa::core::ExecParams* params,
            const core::ConstraintParams* cparams,
            SReal dt,
            sofa::core::MultiVecId pos,
            sofa::core::MultiVecId freePos,
            sofa::core::MultiVecDerivId freeVel,
            simulation::common::MechanicalOperations* mop,
            sofa::core::objectmodel::BaseContext* context,
            sofa::simulation::CpuTask::Status* status,
            bool parallelSolve = false);
    ~FreeMotionTask() override = default;
    sofa::simulation::Task::MemoryAlloc run() final;

private:
    sofa::simulation::Node* m_node;
    const sofa::core::ExecParams* m_params;
    const core::ConstraintParams* m_cparams;
    SReal m_dt;

    sofa::core::MultiVecId m_pos;
    sofa::core::MultiVecId m_freePos;
    sofa::core::MultiVecDerivId m_freeVel;

    simulation::common::MechanicalOperations* m_mop;

    sofa::core::objectmodel::BaseContext* m_context;

    bool m_parallelSolve {false };
};

} // namespace sofa::component::animationloop
