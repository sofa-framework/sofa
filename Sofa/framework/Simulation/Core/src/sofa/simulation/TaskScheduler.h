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

#include <sofa/config.h>

#include <sofa/simulation/Task.h>

#include <string> 
#include <functional>

namespace sofa::simulation
{

/**
 * Base class for a task scheduler
 *
 * The API allows to:
 * - initialize the scheduler with a number of dedicated threads
 * - add a task to the scheduler
 * - wait until all tasks are done etc.
 */
class SOFA_SIMULATION_CORE_API TaskScheduler
{
public:
    virtual ~TaskScheduler() = default;

    /**
    * Assuming 2 concurrent threads by CPU core, return the number of CPU core on the system
    */
    static unsigned GetHardwareThreadsCount();

    // interface
    virtual void init(const unsigned int nbThread = 0) = 0;
            
    virtual void stop(void) = 0;
            
    virtual unsigned int getThreadCount(void) const = 0;

    virtual const char* getCurrentThreadName() = 0;

    virtual int getCurrentThreadType() = 0;

    // queue task if there is space, and run it otherwise
    virtual bool addTask(Task* task) = 0;

    virtual bool addTask(Task::Status& status, const std::function<void()>& task);

    virtual void workUntilDone(Task::Status* status) = 0;

    virtual Task::Allocator* getTaskAllocator() = 0;

protected:

    friend class Task;





public:

    /**
     * Deprecated API. Use TaskSchedulerFactory instead.
     */
    ///@{

        SOFA_ATTRIBUTE_DISABLED_STATIC_TASKSCHEDULER()
        static TaskScheduler* create(const char* name = "");

        SOFA_ATTRIBUTE_DISABLED_STATIC_TASKSCHEDULER()
        typedef std::function<TaskScheduler* ()> TaskSchedulerCreatorFunction;

        SOFA_ATTRIBUTE_DISABLED_STATIC_TASKSCHEDULER()
        static bool registerScheduler(const char* name, TaskSchedulerCreatorFunction creatorFunc);

        SOFA_ATTRIBUTE_DISABLED_STATIC_TASKSCHEDULER()
        static TaskScheduler* getInstance();

        SOFA_ATTRIBUTE_DISABLED_STATIC_TASKSCHEDULER()
        static std::string getCurrentName();

    ///@}
};

} // namespace sofa::simulation
