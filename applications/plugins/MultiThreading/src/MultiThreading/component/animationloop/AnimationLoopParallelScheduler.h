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

#include <MultiThreading/config.h>
#include <sofa/simulation/task/TaskSchedulerUser.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/helper/AdvancedTimer.h>

namespace sofa::simulation
{
class TaskScheduler;
}

namespace multithreading::component::animationloop
{

class SOFA_MULTITHREADING_PLUGIN_API AnimationLoopParallelScheduler :
        public sofa::core::behavior::BaseAnimationLoop,
        public sofa::simulation::TaskSchedulerUser
{
public:

    typedef sofa::core::behavior::BaseAnimationLoop Inherit;
    SOFA_CLASS(AnimationLoopParallelScheduler,sofa::core::behavior::BaseAnimationLoop);

protected:
    AnimationLoopParallelScheduler(sofa::simulation::Node* gnode = NULL);

    ~AnimationLoopParallelScheduler() override;

public:
    void init() override;

    /// Initialization method called at graph creation and modification, during bottom-up traversal.
    void bwdInit() override;

    /// Update method called when variables used in precomputation are modified.
    void reinit() override;

    void cleanup() override;

    void step(const sofa::core::ExecParams* params, SReal dt) override;

private :

    sofa::simulation::Node* gnode;

};
} // namespace multithreading::component::animationloop
