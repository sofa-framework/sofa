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
#include <gtest/gtest.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/CpuTask.h>
#include <sofa/simulation/CpuTaskStatus.h>

#include <atomic>

namespace sofa
{

// Reproduction for a use-after-free in WorkerThread::pushTask. The function
// looks like:
//
//   { lock; ...; m_tasks.push_back(task); }              // task is now visible
//   if (m_taskScheduler->testMainTaskStatus(nullptr)) {
//       m_taskScheduler->setMainTaskStatus(task->getStatus());  // <-- UAF
//       ...
//   }
//
// Once m_tasks.push_back exposes the task to workers and the lock is released,
// any worker can pop, run, and (for tasks that return MemoryAlloc::Dynamic
// from run()) free the task. The post-lock task->getStatus() then reads the
// vtable through freed memory.
//
// The race fires only when:
//   1. testMainTaskStatus(nullptr) is true at the post-lock check, i.e.
//      this is the FIRST push of a workUntilDone cycle (m_mainTaskStatus
//      is set on the first push and cleared by workUntilDone), AND
//   2. the task is short enough that a worker pops, runs, and frees it
//      before the main thread reaches the second task->getStatus() deref.
//
// Hence we tune the test for one task per workUntilDone cycle so every push
// goes through the racy branch, and we run a large number of cycles to make
// the race statistically certain.

namespace
{

void singleShortLambdaCycle(simulation::TaskScheduler& scheduler,
                            std::atomic<int>& counter)
{
    simulation::CpuTaskStatus status;
    scheduler.addTask(status, [&counter]() {
        counter.fetch_add(1, std::memory_order_relaxed);
    });
    scheduler.workUntilDone(&status);
}

} // namespace

// Stress the lambda overload of addTask with one short task per cycle.
// Every push hits the racy branch in pushTask. If the use-after-free is
// present, this test crashes non-deterministically on machines with
// multiple worker threads. On the fixed code, it completes cleanly with
// the expected counter total.
TEST(PushTaskRace, ShortLambdaSingleTaskCycles)
{
    constexpr int kCycles = 500000;

    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(
        simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);

    // Force a real multi-threaded scheduler. The race only exists when
    // pushTask actually queues to a worker (>= 2 threads).
    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; race cannot manifest";
    }

    std::atomic<int> counter { 0 };
    for (int c = 0; c < kCycles; ++c)
    {
        singleShortLambdaCycle(*scheduler, counter);
    }

    EXPECT_EQ(counter.load(std::memory_order_relaxed), kCycles);

    scheduler->stop();
}

} // namespace sofa
