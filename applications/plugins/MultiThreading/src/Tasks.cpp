#include "Tasks.h"

#include "TaskScheduler.h"
//#include "TasksAllocator.h"

#include <assert.h>
#include <thread>
//#include <boost/thread.hpp>

namespace sofa
{

	namespace simulation
	{


		Task::Task(const Task::Status* pStatus)
			: _status(pStatus)
		{
            
		}

		Task::~Task()
		{
//            delete this;
		}
        
        
		ThreadSpecificTaskLockFree::ThreadSpecificTaskLockFree(std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus )
			: Task(pStatus)
			, _atomicCounter(atomicCounter) 
			, _threadSpecificMutex(mutex)
		{}

		ThreadSpecificTaskLockFree::~ThreadSpecificTaskLockFree()
		{
		}

		bool ThreadSpecificTaskLockFree::run(WorkerThread* )
		{  

			runThreadSpecific();

			{
				std::lock_guard<std::mutex> lock(*_threadSpecificMutex);
				runCriticalThreadSpecific();
			}

            _atomicCounter->fetch_sub(1, std::memory_order_acq_rel);

            while(_atomicCounter->load(std::memory_order_relaxed) > 0)
			{  
				// yield while waiting  
				std::this_thread::yield();
			}  
			return false;
		}  

	


	} // namespace simulation

} // namespace sofa
