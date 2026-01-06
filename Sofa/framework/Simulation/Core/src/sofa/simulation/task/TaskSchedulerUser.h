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

#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/core/objectmodel/Base.h>

namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API TaskSchedulerUser : virtual public sofa::core::Base
{
public:
    sofa::Data<int> d_nbThreads;
    sofa::Data<std::string> d_taskSchedulerType; ///< Type of task scheduler to use.

protected:
    sofa::simulation::TaskScheduler* m_taskScheduler { nullptr };

    TaskSchedulerUser();
    void initTaskScheduler();

    void reinitTaskScheduler();
    void stopTaskSchduler();
};

}
