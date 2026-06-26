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
#include <map>
#include <string>
#include <optional>

namespace sofa::simulation
{

class TaskScheduler;

/**
 * Container for task schedulers and its associated name
 * The registry is also owner of the schedulers: it destroys them in its destructor
 */
class SOFA_SIMULATION_CORE_API TaskSchedulerRegistry
{
public:

    /**
     * Add a task scheduler to the registry and transfer the ownership
     */
    bool addTaskSchedulerToRegistry(TaskScheduler* taskScheduler, const std::string& taskSchedulerName);

    /**
     * @return a @TaskScheduler if the scheduler name is found in the registry, nullptr otherwise
     */
    [[nodiscard]] TaskScheduler* getTaskScheduler(const std::string& taskSchedulerName) const;

    /**
     * @return true if the scheduler name is found in the registry, false otherwise
     */
    [[nodiscard]] bool hasScheduler(const std::string& taskSchedulerName) const;

    [[nodiscard]] const std::optional<std::pair<std::string, TaskScheduler*> >& getLastInserted() const;

    /**
     * Clear the registry and destroy the task schedulers sstored in the registry
     */
    void clear();

    ~TaskSchedulerRegistry();

protected:

    std::map<std::string, TaskScheduler*> m_schedulers;
    std::optional<std::pair<std::string, TaskScheduler*> > m_lastInserted {};
};

}
