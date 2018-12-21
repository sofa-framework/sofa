#include "TaskSchedulerTestTasks.h"

#include <sofa/simulation/TaskScheduler.h>

using sofa::simulation::Task;

namespace sofa
{


	bool FibonacciTask::run()
	{
		if (_N < 2)
		{
			*_sum = _N;
			return false;
		}

		Task::Status status;

		int64_t x, y;

        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::getInstance();

		FibonacciTask task0(_N - 1, &x, &status);
		FibonacciTask task1(_N - 2, &y, &status);

        scheduler->addTask(&task0);
        scheduler->addTask(&task1);
        scheduler->workUntilDone(&status);
		
		// Do the sum
		*_sum = x + y;

		return false;
	}



	bool IntSumTask::run()
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

        simulation::TaskScheduler* scheduler = simulation::TaskScheduler::getInstance();

		IntSumTask task0(_first, mid, &x, &status);
		IntSumTask task1(mid+1, _last, &y, &status);

        scheduler->addTask(&task0);
        scheduler->addTask(&task1);
        scheduler->workUntilDone(&status);

		// Do the sum
		*_sum = x + y;


		return false;
	}
	

} // namespace sofa
