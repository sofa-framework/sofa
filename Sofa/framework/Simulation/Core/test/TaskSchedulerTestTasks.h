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
#include <sofa/simulation/CpuTask.h>

namespace sofa
{
    // compute recursively the Fibonacci number for input N  O(~1.6 exp(N))
    // this is implemented to test the task scheduler generating super lightweight tasks and not for performance
    class FibonacciTask : public simulation::CpuTask
    {
    public:
        FibonacciTask(const int64_t N, int64_t* const sum, simulation::CpuTask::Status* status)
        : CpuTask(status)
        , _N(N)
        , _sum(sum)
        {}
        
        ~FibonacciTask() override { }
        
        MemoryAlloc run() final;
        
    private:
        
        const int64_t _N;
        int64_t* const _sum;
    };
    
    
    // compute recursively the sum of integers from first to last
    // this is implemented to test the task scheduler generating super lightweight tasks and not for performance
    class IntSumTask : public simulation::CpuTask
    {
    public:
        IntSumTask(const int64_t first, const int64_t last, int64_t* const sum, simulation::CpuTask::Status* status)
        : CpuTask(status) 
        , _first(first)
        , _last(last)
        , _sum(sum)
        {}
        
        ~IntSumTask() override {}
        
        MemoryAlloc run() final;
        
        
    private:
        
        const int64_t _first;
        const int64_t _last;
        int64_t* const _sum;
        
    };
} // namespace sofa
