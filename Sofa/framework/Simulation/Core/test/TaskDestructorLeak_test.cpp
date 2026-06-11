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
#include <sofa/simulation/task/CpuTask.h>
#include <sofa/simulation/task/CpuTaskStatus.h>
#include <sofa/simulation/task/DefaultTaskScheduler.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>

#include <atomic>
#include <memory>

namespace sofa
{

// Reproduction for a leak in WorkerThread::runTask. After running a task that
// returns MemoryAlloc::Dynamic, the framework calls Task::operator delete to
// release the memory but does NOT call the task's destructor. From
// WorkerThread::runTask:
//
//   if (task->run() & Task::MemoryAlloc::Dynamic) {
//       // pooled memory: call destructor and free
//       //task->~Task();                                    <-- commented out
//       task->operator delete(task, sizeof(*task));
//   }
//
// As a result, any task with non-trivially-destructible members (std::function,
// std::shared_ptr, std::vector, ...) leaks those members' resources every time
// it runs. The std::function-based addTask(Status&, lambda) overload always
// triggers this because CallableTask wraps the lambda in a std::function.
//
// We demonstrate the leak with a custom CpuTask holding a std::shared_ptr.
// If the destructor ran, every task copy of the shared_ptr would release a
// reference and the original's use_count would return to 1 after the burst.
// On the buggy code, use_count stays at 1 + (number of dispatched tasks).

namespace
{

class SharedPtrHoldingTask : public simulation::CpuTask
{
public:
    SharedPtrHoldingTask(simulation::CpuTask::Status* status,
                         std::shared_ptr<int> resource,
                         std::atomic<int>* counter)
        : simulation::CpuTask(status)
        , m_resource(std::move(resource))
        , m_counter(counter)
    {}

    sofa::simulation::Task::MemoryAlloc run() final
    {
        // Touch the resource so the compiler can't elide the member.
        if (m_resource)
        {
            m_counter->fetch_add(*m_resource, std::memory_order_relaxed);
        }
        return sofa::simulation::Task::MemoryAlloc::Dynamic;
    }

private:
    std::shared_ptr<int> m_resource;
    std::atomic<int>* m_counter;
};

} // namespace

// Dispatch many tasks, each holding a copy of the same shared_ptr. After
// workUntilDone, all task instances must have been destroyed; the only
// remaining holder is the test's local `resource`, so use_count must be 1.
//
// On the buggy code, the destructors are skipped and use_count equals
// 1 + kNumTasks.
TEST(TaskDestructorLeak, SharedPtrTasksReleaseTheirReference)
{
    constexpr int kNumTasks = 64;

    auto* scheduler = simulation::MainTaskSchedulerFactory::createInRegistry(
        simulation::DefaultTaskScheduler::name());
    ASSERT_NE(scheduler, nullptr);

    scheduler->init(0);
    if (scheduler->getThreadCount() < 2)
    {
        GTEST_SKIP() << "scheduler has fewer than 2 threads; skipping";
    }

    auto resource = std::make_shared<int>(1);
    std::atomic<int> counter { 0 };

    {
        simulation::CpuTaskStatus status;
        for (int i = 0; i < kNumTasks; ++i)
        {
            // Each task holds its own shared_ptr copy.
            scheduler->addTask(new SharedPtrHoldingTask(&status, resource, &counter));
        }
        scheduler->workUntilDone(&status);
    }

    EXPECT_EQ(counter.load(std::memory_order_relaxed), kNumTasks);
    // After all tasks have been disposed of, only `resource` should hold the
    // shared_ptr. If the framework skipped the destructor, every task's copy
    // is leaked and use_count == 1 + kNumTasks.
    EXPECT_EQ(resource.use_count(), 1);

    scheduler->stop();
}

} // namespace sofa
