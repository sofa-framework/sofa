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
#include <sofa/simulation/TaskScheduler.h>

#include <sofa/simulation/TaskSchedulerFactory.h>

namespace sofa::simulation
{

TaskScheduler* TaskScheduler::create(const char* name)
{
    return TaskSchedulerFactory::create(name);
}

bool TaskScheduler::registerScheduler(const char* name, TaskSchedulerCreatorFunction creatorFunc)
{
    return TaskSchedulerFactory::registerScheduler(name, creatorFunc);
}
        
TaskScheduler* TaskScheduler::getInstance()
{
    return TaskSchedulerFactory::create();
}

std::string TaskScheduler::getCurrentName()
{
    if (const auto& lastCreated = TaskSchedulerFactory::getLastCreated())
    {
        return lastCreated.value().first;
    }

    return {};
}

} // namespace sofa::simulation
