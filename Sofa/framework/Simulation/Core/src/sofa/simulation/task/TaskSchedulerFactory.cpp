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
#include <sofa/simulation/task/TaskSchedulerFactory.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/helper/logging/Messaging.h>

namespace sofa::simulation
{

bool TaskSchedulerFactory::registerScheduler(const std::string& name,
                                             const std::function<TaskScheduler*()>& creatorFunc)
{
    const bool isInserted = m_schedulerCreationFunctions.insert({name, creatorFunc}).second;
    msg_error_when(!isInserted, "TaskSchedulerFactory") << "Cannot register task scheduler '" << name
            << "' into the factory: a task scheduler with this name already exists";
    return isInserted;
}

TaskScheduler* TaskSchedulerFactory::instantiate(const std::string& name)
{
    TaskScheduler* scheduler { nullptr };
    const auto creationIt = m_schedulerCreationFunctions.find(name);
    if (creationIt != m_schedulerCreationFunctions.end())
    {
        scheduler = creationIt->second();
    }
    else
    {
        msg_error("TaskSchedulerFactory") << "Cannot instantiate task scheduler '" << name
            << "': it has not been registered into the factory";
    }
    return scheduler;
}

std::set<std::string> TaskSchedulerFactory::getAvailableSchedulers()
{
    std::set<std::string> schedulers;
    for (const auto& [name, _] : m_schedulerCreationFunctions)
    {
        schedulers.insert(name);
    }
    return schedulers;
}

}
