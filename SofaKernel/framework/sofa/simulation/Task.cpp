#include "Task.h"

//#include "TaskScheduler.h"
//#include "TasksAllocator.h"

#include <assert.h>
#include <thread>


namespace sofa
{
	namespace simulation
	{
        

        Task::Allocator* Task::_allocator = nullptr;


		Task::Task(const Task::Status* status, Thread scheduledThread)
			: _scheduledThread(scheduledThread)
            , _status(status)
            , _id(0)
		{            
		}

		Task::~Task()
		{
		}
        
        
		ThreadSpecificTask::ThreadSpecificTask(std::atomic<int>* atomicCounter, std::mutex* mutex, const Task::Status* status )
			: Task(status)
			, _atomicCounter(atomicCounter) 
			, _threadSpecificMutex(mutex)
		{}

		ThreadSpecificTask::~ThreadSpecificTask()
		{
		}

        Task::MemoryAlloc ThreadSpecificTask::run()
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
			return Task::MemoryAlloc::Stack;
		}  

	
	} // namespace simulation

} // namespace sofa
