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

#include <sofa/simulation/config.h>
#include <functional>
#include <set>
#include <map>
#include <string>

namespace sofa::simulation
{

class TaskScheduler;

/**
 * Simple factory structure used to instantiate a @TaskScheduler based on a name. The name and a
 * creation function must be registered before trying to instantiate.
 */
class SOFA_SIMULATION_CORE_API TaskSchedulerFactory
{
public:

    /**
     * Register a new scheduler in the factory
     *
     * @param name key in the factory
     * @param creatorFunc function creating a new TaskScheduler or a derived class
     * @return false if scheduler could not be registered
     */
    bool registerScheduler(const std::string& name,
                           const std::function<TaskScheduler* ()>& creatorFunc);

    TaskScheduler* instantiate(const std::string& name);

    /**
     * @return a list of registered schedulers
     */
    std::set<std::string> getAvailableSchedulers();

private:
    /// factory map: registered schedulers: name, creation function
    std::map<std::string, std::function<TaskScheduler*()> > m_schedulerCreationFunctions;
};

}
