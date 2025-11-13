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

#include <sofa/simulation/task/TaskSchedulerRegistry.h>
#include <mutex>

namespace sofa::simulation
{

/**
 * A set of static functions with the same interface than a @TaskSchedulerRegistry, working on a
 * single instance of a @TaskSchedulerRegistry.
 * All functions are thread-safe.
 */
class SOFA_SIMULATION_CORE_API MainTaskSchedulerRegistry
{
public:

    static bool addTaskSchedulerToRegistry(TaskScheduler* taskScheduler, const std::string& taskSchedulerName);

    static TaskScheduler* getTaskScheduler(const std::string& taskSchedulerName);

    static bool hasScheduler(const std::string& taskSchedulerName);

    static const std::optional<std::pair<std::string, TaskScheduler*> >& getLastInserted();

    static void clear();

private:
    static std::mutex s_mutex;

    static TaskSchedulerRegistry& getInstance();
};

}
