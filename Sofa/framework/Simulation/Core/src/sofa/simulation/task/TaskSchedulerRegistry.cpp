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
#include <sofa/simulation/task/TaskSchedulerRegistry.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/simulation/task/TaskScheduler.h>

namespace sofa::simulation
{

bool TaskSchedulerRegistry::addTaskSchedulerToRegistry(TaskScheduler* taskScheduler, const std::string& taskSchedulerName)
{
    const auto [fst, snd] = m_schedulers.insert({taskSchedulerName, taskScheduler});
    msg_error_when(!snd, "TaskSchedulerRegistry") << "Cannot insert task scheduler '" << taskSchedulerName
            << "' in the registry: a task scheduler with this name already exists";

    if (snd)
    {
        m_lastInserted = std::make_pair(taskSchedulerName, taskScheduler);
    }
    else
    {
        m_lastInserted.reset();
    }

    return snd;
}

TaskScheduler* TaskSchedulerRegistry::getTaskScheduler(const std::string& taskSchedulerName) const
{
    const auto it = m_schedulers.find(taskSchedulerName);
    if (it != m_schedulers.end())
    {
        return it->second;
    }
    return nullptr;
}

bool TaskSchedulerRegistry::hasScheduler(const std::string& taskSchedulerName) const
{
    return m_schedulers.contains(taskSchedulerName);
}

const std::optional<std::pair<std::string, TaskScheduler*>>& TaskSchedulerRegistry::getLastInserted() const
{
    return m_lastInserted;
}

void TaskSchedulerRegistry::clear()
{
    for (const auto& p : m_schedulers)
    {
        delete p.second;
    }
    m_schedulers.clear();
}

TaskSchedulerRegistry::~TaskSchedulerRegistry()
{
    clear();
}

}
