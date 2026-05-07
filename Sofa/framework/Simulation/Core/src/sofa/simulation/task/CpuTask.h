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

#include <sofa/simulation/task/CpuTaskStatus.h>


namespace sofa::simulation
{
/**  Base class to implement a CPU task
 *   all the tasks running on the CPU should inherits from this class
 */
class SOFA_SIMULATION_CORE_API CpuTask : public Task
{
public:

    using Status = CpuTaskStatus;

    Status* getStatus(void) const override final;


    CpuTask(Status* status, int scheduledThread = -1);

    virtual ~CpuTask() = default;

private:
    Status* m_status { nullptr };
};

} // namespace sofa::simulation
