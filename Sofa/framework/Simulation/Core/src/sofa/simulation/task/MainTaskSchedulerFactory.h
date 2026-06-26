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

#include <sofa/simulation/task/TaskSchedulerFactory.h>
#include <mutex>

namespace sofa::simulation
{

/**
 * A set of static function with the same interface than @TaskSchedulerFactory, working on a single
 * instance of @TaskSchedulerFactory.
 *
 * The static functions @createInRegistry use the factory to instantiate a task scheduler
 * and store it in @MainTaskSchedulerRegistry
 */
class SOFA_SIMULATION_CORE_API MainTaskSchedulerFactory
{
public:

    static bool registerScheduler(const std::string& name,
                                  const std::function<TaskScheduler* ()>& creatorFunc);

    static TaskScheduler* instantiate(const std::string& name);

    static std::set<std::string> getAvailableSchedulers();


    static TaskScheduler* createInRegistry(const std::string& name);
    static TaskScheduler* createInRegistry();

    static std::string defaultTaskSchedulerType();

private:
    static std::mutex s_mutex;

    static TaskSchedulerFactory& getFactory();
};

}
