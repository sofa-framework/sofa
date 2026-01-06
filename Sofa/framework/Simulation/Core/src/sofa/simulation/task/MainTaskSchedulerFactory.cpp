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
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/MainTaskSchedulerRegistry.h>
#include <sofa/simulation/task/Task.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/task/DefaultTaskScheduler.h>

namespace sofa::simulation
{

std::mutex MainTaskSchedulerFactory::s_mutex;

bool MainTaskSchedulerFactory::registerScheduler(const std::string& name,
    const std::function<TaskScheduler*()>& creatorFunc)
{
    std::lock_guard lock(s_mutex);
    return getFactory().registerScheduler(name, creatorFunc);
}

TaskScheduler* MainTaskSchedulerFactory::createInRegistry(const std::string& name)
{
    std::lock_guard lock(s_mutex);

    TaskScheduler* scheduler = MainTaskSchedulerRegistry::getTaskScheduler(name);
    if (scheduler == nullptr)
    {
        scheduler = getFactory().instantiate(name);

        if (scheduler)
        {
            MainTaskSchedulerRegistry::addTaskSchedulerToRegistry(scheduler, name);
        }
    }

    if (scheduler)
    {
        Task::setAllocator(scheduler->getTaskAllocator());
    }

    return scheduler;
}

TaskScheduler* MainTaskSchedulerFactory::createInRegistry()
{
    return createInRegistry(defaultTaskSchedulerType());
}

std::string MainTaskSchedulerFactory::defaultTaskSchedulerType()
{
    return DefaultTaskScheduler::name();
}

TaskScheduler* MainTaskSchedulerFactory::instantiate(const std::string& name)
{
    std::lock_guard lock(s_mutex);
    return getFactory().instantiate(name);
}

std::set<std::string> MainTaskSchedulerFactory::getAvailableSchedulers()
{
    std::lock_guard lock(s_mutex);
    return getFactory().getAvailableSchedulers();
}

TaskSchedulerFactory& MainTaskSchedulerFactory::getFactory()
{
    static TaskSchedulerFactory f;
    return f;
}

}
