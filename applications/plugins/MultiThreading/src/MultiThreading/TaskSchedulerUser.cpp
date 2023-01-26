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
#include <MultiThreading/TaskSchedulerUser.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

namespace multithreading
{

void TaskSchedulerUser::initTaskScheduler()
{
    m_taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry();
    assert(m_taskScheduler != nullptr);
    if (m_taskScheduler->getThreadCount() < 1)
    {
        auto nbThreads = sofa::helper::getWriteAccessor(d_nbThreads);
        if (nbThreads <= 0)
        {
            const auto nbCPUCores = static_cast<int>(sofa::simulation::TaskScheduler::GetHardwareThreadsCount());
            nbThreads.wref() = nbCPUCores - nbThreads;
            nbThreads.wref() = std::max(0, nbThreads.ref());
        }

        m_taskScheduler->init(nbThreads);
        msg_info() << "Task scheduler initialized on " << m_taskScheduler->getThreadCount() << " threads";
    }
    else
    {
        msg_info() << "Task scheduler already initialized on " << m_taskScheduler->getThreadCount() << " threads";
    }
}

TaskSchedulerUser::TaskSchedulerUser()
    : d_nbThreads(initData(&d_nbThreads, 0, "nbThreads",
"If not yet initialized, the main task scheduler is initialized with this number of threads. "
    "0 corresponds to the number of available cores on the CPU. "
    "-n (minus) corresponds to the number of available cores on the CPU minus the provided number."))
{
}

}
