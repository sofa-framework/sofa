#include "TaskSchedulerTestTasks.h"

#include <MultiThreading/src/TaskScheduler.h>
#include <MultiThreading/src/DefaultTaskScheduler.h>
#include <sofa/helper/testing/BaseTest.h>

namespace sofa
{

	// compute the Fibonacci number for input N
	static int64_t Fibonacci(int64_t N, int nbThread = 0)
	{
        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::create(simulation::TaskSchedulerDefault::name());
        scheduler->init(nbThread);

        simulation::Task::Status status;
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
        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::create(simulation::TaskSchedulerDefault::name());
        scheduler->init(nbThread);

        simulation::Task::Status status;
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
