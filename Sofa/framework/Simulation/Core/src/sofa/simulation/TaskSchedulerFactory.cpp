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
#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/TaskSchedulerFactory.h>
#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/DefaultTaskScheduler.h>

namespace sofa::simulation
{

std::map<std::string, std::function<TaskScheduler*()>> TaskSchedulerFactory::s_schedulerCreationFunctions;
std::map<std::string, std::unique_ptr<TaskScheduler> > TaskSchedulerFactory::s_schedulers;

bool TaskSchedulerFactory::registerScheduler(const std::string& name,
    std::function<TaskScheduler*()> creatorFunc)
{
    const bool isInserted = s_schedulerCreationFunctions.insert({name, creatorFunc}).second;
    msg_error_when(!isInserted, "TaskSchedulerFactory") << "Cannot register task scheduler '" << name
            << "' into the factory: a task scheduler with this name already exists";
    return isInserted;
}

TaskScheduler* TaskSchedulerFactory::create(const std::string& name)
{
    TaskScheduler* scheduler { nullptr };
    if (const auto it = s_schedulers.find(name); it != s_schedulers.end())
    {
        scheduler = it->second.get();
    }
    else
    {
        const auto creationIt = s_schedulerCreationFunctions.find(name);
        if (creationIt != s_schedulerCreationFunctions.end())
        {
            const auto [fst, snd] = s_schedulers.insert(
                {name, std::unique_ptr<TaskScheduler>(creationIt->second())});

            msg_error_when(!snd, "TaskSchedulerFactory") << "Cannot create task scheduler '" << name
                    << "' from the factory: a task scheduler with this name already exists";

            scheduler = fst->second.get();
        }
        else
        {
            msg_error("TaskSchedulerFactory") << "Cannot create task scheduler '" << name
                << "': it has not been registered into the factory";
        }
    }

    if (scheduler)
    {
        Task::setAllocator(scheduler->getTaskAllocator());
        s_lastCreated = std::make_pair(name, scheduler);
    }
    else
    {
        s_lastCreated.reset();
    }

    return scheduler;
}

TaskScheduler* TaskSchedulerFactory::create()
{
    return TaskSchedulerFactory::create(DefaultTaskScheduler::name());
}

const std::optional<std::pair<std::string, TaskScheduler*> >& TaskSchedulerFactory::getLastCreated()
{
    return s_lastCreated;
}

std::set<std::string> TaskSchedulerFactory::getAvailableSchedulers()
{
    std::set<std::string> schedulers;
    for (const auto& [name, _] : s_schedulerCreationFunctions)
    {
        schedulers.insert(name);
    }
    return schedulers;
}

void TaskSchedulerFactory::clear()
{
    s_schedulers.clear();
}

}
