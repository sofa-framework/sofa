#include "TaskSchedulerTestTasks.h"

#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/DefaultTaskScheduler.h>
#include <sofa/helper/testing/BaseTest.h>

namespace sofa
{
    TEST(TaskSchedulerTests, DefaultTaskScheduler)
    {
        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::create(simulation::DefaultTaskScheduler::name());
        EXPECT_NE(scheduler, nullptr);
        //Make sure that the created TaskScheduler is of derived class DefaultTaskScheduler
        EXPECT_NE(dynamic_cast<simulation::DefaultTaskScheduler*>(scheduler), nullptr);

        //The default name is "Main  0"
        EXPECT_EQ(std::string(scheduler->getCurrentThreadName()), "Main  0");

        // scheduler has not been initialized yet
        EXPECT_EQ(scheduler->getThreadCount(), 0);

        scheduler->init(0);
        EXPECT_EQ(scheduler->getThreadCount(), std::thread::hardware_concurrency() / 2);

        //Create a scheduler with another name. It has consequences on the previously created task scheduler
        simulation::TaskScheduler* scheduler_2 = simulation::TaskScheduler::create("notRegisteredName");
        EXPECT_NE(scheduler_2, nullptr);
        EXPECT_EQ(scheduler, scheduler_2);
        EXPECT_NE(dynamic_cast<simulation::DefaultTaskScheduler*>(scheduler_2), nullptr);

        //creating a "new" DefaultTaskScheduler uninitializes the previous instance
        EXPECT_EQ(scheduler->getThreadCount(), 0);
        EXPECT_EQ(scheduler_2->getThreadCount(), 0);

        EXPECT_EQ(simulation::TaskScheduler::getInstance(), scheduler_2);

        simulation::TaskScheduler* scheduler_3 = simulation::TaskScheduler::create(simulation::DefaultTaskScheduler::name());
        EXPECT_NE(scheduler_3, nullptr);
        EXPECT_EQ(scheduler, scheduler_3);
        EXPECT_NE(dynamic_cast<simulation::DefaultTaskScheduler*>(scheduler_3), nullptr);

        EXPECT_EQ(scheduler->getThreadCount(), 0);
        EXPECT_EQ(scheduler_2->getThreadCount(), 0);
        EXPECT_EQ(scheduler_3->getThreadCount(), 0);

        scheduler->init(0);

        simulation::CpuTask::Status status;
        SleepTask sleepTask(&status);
        EXPECT_FALSE(sleepTask.m_isTaskDone);
        scheduler->addTask(&sleepTask);
        EXPECT_FALSE(sleepTask.m_isTaskDone);

        simulation::TaskScheduler* scheduler_4 = simulation::TaskScheduler::create("anotherName");
        EXPECT_NE(scheduler_4, nullptr);

        scheduler->workUntilDone(&status);
        EXPECT_TRUE(sleepTask.m_isTaskDone);

        EXPECT_NE(scheduler_4, nullptr);

        scheduler->init(0);
        EXPECT_EQ(scheduler->getThreadCount(), std::thread::hardware_concurrency() / 2);

        simulation::CpuTask::Status threadIdTaskStatus;

        const unsigned int nbTasks = scheduler->getThreadCount() * 5;
        std::vector<std::thread::id> threadIds(nbTasks);
        std::vector<ThreadIdTask> tasks;
        for (unsigned int i = 0; i < nbTasks; ++i)
        {
            tasks.emplace_back(&threadIds[i], &threadIdTaskStatus);
            auto& lastTask = tasks.back();
            EXPECT_FALSE(lastTask.m_isTaskDone);
        }
        for (auto& task : tasks)
        {
            scheduler->addTask(&task);
        }
        scheduler->workUntilDone(&threadIdTaskStatus);
        EXPECT_FALSE(threadIdTaskStatus.isBusy());

        std::set<std::thread::id> uniqueThreadIds(std::cbegin(threadIds), std::cend(threadIds));
        EXPECT_EQ(uniqueThreadIds.size(), std::thread::hardware_concurrency() / 2);

        scheduler->stop();
    }

    // compute the Fibonacci number for input N
    static int64_t Fibonacci(int64_t N, int nbThread = 0)
    {
        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::create(simulation::DefaultTaskScheduler::name());
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
        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::create(simulation::DefaultTaskScheduler::name());
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
        int64_t res = IntSum1ToN(N, 1);
        EXPECT_EQ(res, (N)*(N+1)/2);
        return;
    }
    
    // compute the sum of integers from 1 to N multi thread
    TEST(TaskSchedulerTests, IntSumMulti)
    {
        const int64_t N = 1 << 20;
        int64_t res = IntSum1ToN(N);
        EXPECT_EQ(res, (N)*(N + 1) / 2);
        return;
    }
    

} // namespace sofa
