//
// Created by paul on 20/02/2026.
//

#pragma once

#include <sofa/simulation/config.h>

#include <sofa/simulation/task/TaskSchedulerUser.h>
#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa::simulation
{

class TaskSchedulerSettings : public core::objectmodel::BaseObject,
                              public TaskSchedulerUser
{
public:
    SOFA_CLASS2(TaskSchedulerSettings, core::objectmodel::BaseObject, TaskSchedulerUser);

    TaskSchedulerSettings() = default;

    void init()
    {
        initTaskScheduler();
    }

    void reInit()
    {
        reinitTaskScheduler();
    }
};


}
