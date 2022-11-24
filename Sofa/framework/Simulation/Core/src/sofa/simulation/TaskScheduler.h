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

#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <map>
#include <string> 
#include <functional>

namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API TaskScheduler
{
            
public:   
    virtual ~TaskScheduler() = default;

    /**
        * Check if a TaskScheduler already exists with this name.
        * If not, it creates and registers a new TaskScheduler of type DefaultTaskScheduler with
        * name as a key
        *
        * @param name key to find or create a TaskScheduler
        * @return A TaskScheduler
        */
    static TaskScheduler* create(const char* name = "");
            
    typedef std::function<TaskScheduler* ()> TaskSchedulerCreatorFunction;

    /**
        * Register a new scheduler in the factory
        *
        * @param name key in the factory
        * @param creatorFunc function creating a new TaskScheduler or a derived class
        * @return
        */
    static bool registerScheduler(const char* name, TaskSchedulerCreatorFunction creatorFunc);

    /**
        * Get the current TaskScheduler instance.
        *
        * If not instance has been created yet, a new one with empty name is created.
        * @return The current TaskScheduler instance
        */
    static TaskScheduler* getInstance();

    /**
        * Get the name of the current TaskScheduler instance
        * @return The name of the current TaskScheduler instance
        */
    static const std::string& getCurrentName()  { return _currentSchedulerName; }
            
    // interface
    virtual void init(const unsigned int nbThread = 0) = 0;
            
    virtual void stop(void) = 0;
            
    virtual unsigned int getThreadCount(void) const = 0;
            
    virtual const char* getCurrentThreadName() = 0;
            
    virtual int getCurrentThreadType() = 0;
            
    // queue task if there is space, and run it otherwise
    virtual bool addTask(Task* task) = 0;
            
    virtual void workUntilDone(Task::Status* status) = 0;
            
    virtual Task::Allocator* getTaskAllocator() = 0;
            
            
protected:
            
    // factory map: registered schedulers: name, creation function
    static std::map<std::string, std::function<TaskScheduler*()> > _schedulers;
            
    // current instantiated scheduler
    static std::string _currentSchedulerName;
    static TaskScheduler* _currentScheduler;
            
    friend class Task;
};
        


} // namespace sofa::simulation
