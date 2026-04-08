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
#include "TaskSchedulerTestTasks.h"

#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

using sofa::simulation::Task;

namespace sofa
{

    
    Task::MemoryAlloc FibonacciTask::run()
    {
        if (_N < 2)
        {
            *_sum = _N;
            return MemoryAlloc::Stack;
        }
        
        simulation::CpuTask::Status status;
        
        int64_t x, y;
        
        simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
        
        FibonacciTask task0(_N - 1, &x, &status);
        FibonacciTask task1(_N - 2, &y, &status);
        
        scheduler->addTask(&task0);
        scheduler->addTask(&task1);
        scheduler->workUntilDone(&status);
        
        // Do the sum
        *_sum = x + y;
        
        return MemoryAlloc::Stack;
    }
    
    
    
    Task::MemoryAlloc IntSumTask::run()
    {
        const int64_t count = _last - _first;
        if (count < 1)
        {
            *_sum = _first;
            return MemoryAlloc::Stack;
        }
        
        const int64_t mid = _first + (count / 2);
        
        simulation::CpuTask::Status status;
        
        int64_t x, y;
        
        simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry();
        
        IntSumTask task0(_first, mid, &x, &status);
        IntSumTask task1(mid+1, _last, &y, &status);
        
        scheduler->addTask(&task0);
        scheduler->addTask(&task1);
        scheduler->workUntilDone(&status);
        
        // Do the sum
        *_sum = x + y;
        
        
        return MemoryAlloc::Stack;
    }
} // namespace sofa
