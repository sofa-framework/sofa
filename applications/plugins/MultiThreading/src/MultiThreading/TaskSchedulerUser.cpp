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
#include <sofa/simulation/InitTasks.h>

namespace multithreading
{

void TaskSchedulerUser::initTaskScheduler()
{
    if (!d_taskSchedulerType.isSet() || d_taskSchedulerType.getValue().empty())
    {
        d_taskSchedulerType.setValue(sofa::simulation::MainTaskSchedulerFactory::defaultTaskSchedulerType());
    }

    m_taskScheduler = sofa::simulation::MainTaskSchedulerFactory::createInRegistry(d_taskSchedulerType.getValue());
    if (m_taskScheduler == nullptr)
    {
        msg_error() << "Could not create task scheduler of type '" << d_taskSchedulerType.getValue() << "'";
        d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

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

void TaskSchedulerUser::reinitTaskScheduler()
{
    if (m_taskScheduler)
    {
        const auto nbThreads = d_nbThreads.getValue();
        if ( nbThreads != static_cast<int>(m_taskScheduler->getThreadCount()) )
        {
            m_taskScheduler->init(nbThreads);
            sofa::simulation::initThreadLocalData();
        }
    }
}

void TaskSchedulerUser::stopTaskSchduler()
{
    if (m_taskScheduler)
    {
        m_taskScheduler->stop();
    }
}

TaskSchedulerUser::TaskSchedulerUser()
    : d_nbThreads(initData(&d_nbThreads, 0, "nbThreads",
"If not yet initialized, the main task scheduler is initialized with this number of threads. "
    "0 corresponds to the number of available cores on the CPU. "
    "-n (minus) corresponds to the number of available cores on the CPU minus the provided number."))
    , d_taskSchedulerType(initData(&d_taskSchedulerType, sofa::simulation::MainTaskSchedulerFactory::defaultTaskSchedulerType(), "taskSchedulerType", "Type of task scheduler to use."))
{
}

}
