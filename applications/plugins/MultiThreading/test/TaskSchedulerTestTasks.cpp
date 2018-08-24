#include "TaskSchedulerTestTasks.h"

#include <MultiThreading/src/TaskScheduler.h>

using sofa::simulation::Task;

namespace sofa
{


	bool FibonacciTask::run(simulation::WorkerThread*)
	{
		if (_N < 2)
		{
			*_sum = _N;
			return false;
		}

		Task::Status status;

		int64_t x, y;

		simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();

		FibonacciTask task0(_N - 1, &x, &status);
		FibonacciTask task1(_N - 2, &y, &status);

		thread->addTask(&task0);
		thread->addTask(&task1);
		thread->workUntilDone(&status);
		
		// Do the sum
		*_sum = x + y;

		return false;
	}



	bool IntSumTask::run(simulation::WorkerThread*)
	{
		const int64_t count = _last - _first;
		if (count < 1)
		{
			*_sum = _first;
			return false;
		}

		const int64_t mid = _first + (count / 2);

		Task::Status status;

		int64_t x, y;

		simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();

		IntSumTask task0(_first, mid, &x, &status);
		IntSumTask task1(mid+1, _last, &y, &status);

		thread->addTask(&task0);
		thread->addTask(&task1);
		thread->workUntilDone(&status);

		// Do the sum
		*_sum = x + y;


		return false;
	}
	

} // namespace sofa