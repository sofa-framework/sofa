#include "TestTasks.h"

#include <MultiThreading/src/TaskScheduler.h>
#include <sofa/helper/testing/BaseTest.h>

namespace sofa
{

	// compute the Fibonacci number for input N
	static int64_t Fibonacci(int64_t N, int nbThread = 0)
	{
		simulation::TaskScheduler::getInstance().init(nbThread);

		simulation::Task::Status status;
		int64_t result = 0;

		simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();
		FibonacciTask task(N, &result, &status);
		thread->addTask(&task);
		thread->workUntilDone(&status);

		simulation::TaskScheduler::getInstance().stop();
		return result;
	}


	// compute the sum of integers from 1 to N
	static int64_t IntSum1ToN(const int64_t N, int nbThread = 0)
	{
		simulation::TaskScheduler::getInstance().init(nbThread);

		simulation::Task::Status status;
		int64_t result = 0;

		simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();
		IntSumTask task(1, N, &result, &status);
		thread->addTask(&task);
		thread->workUntilDone(&status);

		simulation::TaskScheduler::getInstance().stop();
		return result;
	}



	// compute the Fibonacci single thread
	TEST(TaskSchedulerTests, FibonacciSingle )
	{ 
		// tested with
		// 13 : 233
		// 23 : 28657
		// 35 : 9227465
		// 41 : 165580141 
		// 43 : 433494437
		// 47 : 2971215073
		int64_t res = Fibonacci(13,1);
		EXPECT_EQ(res, 233);
		return;
	}

	// compute the Fibonacci multi thread
	TEST(TaskSchedulerTests, FibonacciMulti)
	{ 
		int64_t res = Fibonacci(13);
		EXPECT_EQ(res, 233);
		return;
	}

	// compute the sum of integers from 1 to N single thread
	TEST(TaskSchedulerTests, IntSumSingle)
	{
		const int64_t N = 1 << 13;
		int64_t res = IntSum1ToN( N, 1);
		EXPECT_EQ(res, (N)*(N+1)/2);
		return;
	}

	// compute the sum of integers from 1 to N multi thread
	TEST(TaskSchedulerTests, IntSumMulti)
	{
		const int64_t N = 1 << 13;
		int64_t res = IntSum1ToN(N);
		EXPECT_EQ(res, (N)*(N + 1) / 2);
		return;
	}


} // namespace sofa
