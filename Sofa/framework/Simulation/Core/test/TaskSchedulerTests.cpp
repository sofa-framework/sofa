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

#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/CpuTask.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/testing/BaseTest.h>

namespace sofa
{
    // compute the Fibonacci number for input N
    static int64_t Fibonacci(int64_t N, int nbThread = 0)
    {
        simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(simulation::DefaultTaskScheduler::name());
        scheduler->init(nbThread);
        
        simulation::CpuTask::Status status;
        int64_t result = 0;
        
        FibonacciTask task(N, &result, &status);
        scheduler->addTask(&task);
        scheduler->workUntilDone(&status);
        
        scheduler->stop();
        return result;
    }
    
    
    // compute the sum of integers from 1 to N
    static int64_t IntSum1ToN(const int64_t N, int nbThread = 0)
    {
        simulation::TaskScheduler* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(simulation::DefaultTaskScheduler::name());
        scheduler->init(nbThread);
        
        simulation::CpuTask::Status status;
        int64_t result = 0;
        
        IntSumTask task(1, N, &result, &status);
        scheduler->addTask(&task);
        scheduler->workUntilDone(&status);
        
        scheduler->stop();
        return result;
    }
    
    
    
    // compute the Fibonacci single thread
    TEST(TaskSchedulerTests, FibonacciSingle )
    { 
        // tested with
        //  3 : 2
        //  6 : 8
        // 13 : 233
        // 23 : 28657
        // 27 : 196418
        // 35 : 9227465
        // 41 : 165580141
        // 43 : 433494437
        // 47 : 2971215073
        const int64_t res = Fibonacci(27, 1);
        EXPECT_EQ(res, 196418);
        return;
    }
    
    // compute the Fibonacci multi thread
    TEST(TaskSchedulerTests, FibonacciMulti)
    {
        // tested with
        //  3 : 2
        //  6 : 8
        // 13 : 233
        // 23 : 28657
        // 27 : 196418
        // 35 : 9227465
        // 41 : 165580141
        // 43 : 433494437
        // 47 : 2971215073
        const int64_t res = Fibonacci(27);
        EXPECT_EQ(res, 196418);
        return;
    }
    
    // compute the sum of integers from 1 to N single thread
    TEST(TaskSchedulerTests, IntSumSingle)
    {
        const int64_t N = 1 << 20;
        const int64_t res = IntSum1ToN(N, 1);
        EXPECT_EQ(res, (N)*(N+1)/2);
        return;
    }
    
    // compute the sum of integers from 1 to N multi thread
    TEST(TaskSchedulerTests, IntSumMulti)
    {
        const int64_t N = 1 << 20;
        const int64_t res = IntSum1ToN(N);
        EXPECT_EQ(res, (N)*(N + 1) / 2);
        return;
    }

    TEST(TaskSchedulerTests, Lambda)
    {
        const auto scheduler = std::unique_ptr<simulation::TaskScheduler>(
            simulation::MainTaskSchedulerFactory::instantiate(simulation::DefaultTaskScheduler::name()));
        scheduler->init(1);

        unsigned int one = 0u;

        simulation::CpuTaskStatus status;
        scheduler->addTask(status, [&one]{ one = 1u; });

        scheduler->workUntilDone(&status);
        scheduler->stop();

        EXPECT_EQ(one, 1u);
    }

    TEST(TaskSchedulerTests, Functor)
    {
        struct Functor
        {
            Functor(unsigned int& num) : m_num(num) {}
            void operator()() const
            {
                m_num = 1u ;
            }
            unsigned int& m_num;
        };

        const auto scheduler = std::unique_ptr<simulation::TaskScheduler>(
            simulation::MainTaskSchedulerFactory::instantiate(simulation::DefaultTaskScheduler::name()));
        scheduler->init(1);

        unsigned int one = 0u;

        simulation::CpuTaskStatus status;
        scheduler->addTask(status, Functor(one));

        scheduler->workUntilDone(&status);
        scheduler->stop();

        EXPECT_EQ(one, 1u);
    }

} // namespace sofa
