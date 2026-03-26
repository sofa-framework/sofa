//
// Created by paul on 20/02/2026.
//

#pragma once

#include <sofa/simulation/config.h>

#include <sofa/simulation/task/TaskSchedulerUser.h>
#include <sofa/core/objectmodel/BaseComponent.h>

namespace sofa::simulation
{

class TaskSchedulerSettings : public core::objectmodel::BaseComponent,
                              public TaskSchedulerUser
{
public:
    SOFA_CLASS2(TaskSchedulerSettings, core::objectmodel::BaseComponent, TaskSchedulerUser);

    TaskSchedulerSettings() = default;

    void init()
    {
        initTaskScheduler(true);
    }

    void reInit()
    {
        reinitTaskScheduler();
    }
};


}
