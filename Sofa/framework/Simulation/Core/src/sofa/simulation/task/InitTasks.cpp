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
#include <sofa/simulation/task/InitTasks.h>

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/simulation/task/TaskScheduler.h>

#include <thread>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>

namespace sofa::simulation
{
InitPerThreadDataTask::InitPerThreadDataTask(std::atomic<int>* atomicCounter, std::mutex* mutex, CpuTask::Status* status)
: CpuTask(status), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
{}
        
Task::MemoryAlloc InitPerThreadDataTask::run()
{
            
    sofa::core::execparams::defaultInstance();

    sofa::core::constraintparams::defaultInstance();

    sofa::core::mechanicalparams::defaultInstance();

    sofa::core::visual::visualparams::defaultInstance();
            
    {
        // to solve IdFactory<Base>::getID() problem in AdvancedTimer functions
        std::lock_guard<std::mutex> lock(*IdFactorygetIDMutex);
                
        helper::AdvancedTimer::begin("Animate");
        helper::AdvancedTimer::end("Animate");
    }
            
    _atomicCounter->fetch_sub(1, std::memory_order_acq_rel);
            
    while (_atomicCounter->load(std::memory_order_relaxed) > 0)
    {
        // yield while waiting  
        std::this_thread::yield();
    }
    return Task::MemoryAlloc::Dynamic;
}
        
        
// temp remove this function to use the global one
void initThreadLocalData()
{
    TaskScheduler* scheduler = MainTaskSchedulerFactory::createInRegistry();
    std::atomic<int> atomicCounter = scheduler->getThreadCount();
            
    std::mutex  InitThreadSpecificMutex;
            
    CpuTask::Status status;
    const int nbThread = scheduler->getThreadCount();
            
    for (int i = 0; i<nbThread; ++i)
    {
        scheduler->addTask(new InitPerThreadDataTask(&atomicCounter, &InitThreadSpecificMutex, &status));
    }
            
    scheduler->workUntilDone(&status);
            
    return;
}
        
} // namespace sofa::simulation
