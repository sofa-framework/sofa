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
#include <sofa/simulation/task/MainTaskSchedulerRegistry.h>

namespace sofa::simulation
{
std::mutex MainTaskSchedulerRegistry::s_mutex;

bool MainTaskSchedulerRegistry::addTaskSchedulerToRegistry(TaskScheduler* taskScheduler,
    const std::string& taskSchedulerName)
{
    std::lock_guard lock(s_mutex);
    return getInstance().addTaskSchedulerToRegistry(taskScheduler, taskSchedulerName);
}

TaskScheduler* MainTaskSchedulerRegistry::getTaskScheduler(const std::string& taskSchedulerName)
{
    std::lock_guard lock(s_mutex);
    return getInstance().getTaskScheduler(taskSchedulerName);
}

bool MainTaskSchedulerRegistry::hasScheduler(const std::string& taskSchedulerName)
{
    std::lock_guard lock(s_mutex);
    return getInstance().hasScheduler(taskSchedulerName);
}

const std::optional<std::pair<std::string, TaskScheduler*>>& MainTaskSchedulerRegistry::
getLastInserted()
{
    std::lock_guard lock(s_mutex);
    return getInstance().getLastInserted();
}

void MainTaskSchedulerRegistry::clear()
{
    std::lock_guard lock(s_mutex);
    getInstance().clear();
}

TaskSchedulerRegistry& MainTaskSchedulerRegistry::getInstance()
{
    static TaskSchedulerRegistry r;
    return r;
}


}

