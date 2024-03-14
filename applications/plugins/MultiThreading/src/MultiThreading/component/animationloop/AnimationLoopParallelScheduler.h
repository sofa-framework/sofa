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
#include <MultiThreading/TaskSchedulerUser.h>
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
        public TaskSchedulerUser
{
public:

    typedef sofa::core::behavior::BaseAnimationLoop Inherit;
    SOFA_CLASS(AnimationLoopParallelScheduler,sofa::core::behavior::BaseAnimationLoop);

    SOFA_ATTRIBUTE_DISABLED__TASKSCHEDULERUSER_DATANAME("Use TaskSchedulerUser::d_taskSchedulerType instead.")
    sofa::core::objectmodel::lifecycle::RemovedData schedulerName {this, "v23.06", "v23.12",
                                                                   "scheduler",
                                                                   "To fix you scene you can rename 'scheduler' with 'taskSchedulerType'."};

    SOFA_ATTRIBUTE_DISABLED__TASKSCHEDULERUSER_DATANAME("Use TaskSchedulerUser::d_nbThreads instead.")
    sofa::core::objectmodel::lifecycle::RemovedData threadNumber {this, "v23.06", "v23.12",
                                                                  "threadNumber",
                                                                  "To fix you scene you can rename 'threadNumber' with 'nbThreads'."};

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

    /// Construction method called by ObjectFactory.
    template<class T>
    static typename T::SPtr create(T*, sofa::core::objectmodel::BaseContext* context, sofa::core::objectmodel::BaseObjectDescription* arg)
    {
        sofa::simulation::Node* gnode = dynamic_cast<sofa::simulation::Node*>(context);
        typename T::SPtr obj = sofa::core::objectmodel::New<T>(gnode);
        if (context) context->addObject(obj);
        if (arg) obj->parse(arg);
        return obj;
    }

private :

    sofa::simulation::Node* gnode;

};
} // namespace multithreading::component::animationloop
