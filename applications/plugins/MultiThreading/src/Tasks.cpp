#include "Tasks.h"

#include <boost/thread.hpp>

namespace sofa
{

	namespace simulation
	{


		Task::Task(const Task::Status* pStatus) 
			: m_Status(pStatus)
		{
		}

		Task::~Task()
		{
			//m_Status;
			//delete this;
		}



		//InitPerThreadDataTask::InitPerThreadDataTask(volatile long* atomicCounter, boost::mutex* mutex, TaskStatus* pStatus ) 
		ThreadSpecificTask::ThreadSpecificTask(helper::system::atomic<int>* atomicCounter, boost::mutex* mutex, Task::Status* pStatus ) 
			: Task(pStatus)
			, mAtomicCounter(atomicCounter) 
			, mThreadSpecificMutex(mutex)
		{}

		ThreadSpecificTask::~ThreadSpecificTask()
		{
			//mAtomicCounter;
		}

		bool ThreadSpecificTask::run(WorkerThread* )
		{  

			runThreadSpecific();


			{
				boost::lock_guard<boost::mutex> lock(*mThreadSpecificMutex);

				runCriticalThreadSpecific();

			}

			//BOOST_INTERLOCKED_DECREMENT( mAtomicCounter );
			//BOOST_COMPILER_FENCE;

			--(*mAtomicCounter);


			while(mAtomicCounter->operator int() > 0)  
			{  
				// yield while waiting  
				boost::this_thread::yield();
			}  
			return false;
		}  

	


	} // namespace simulation

} // namespace sofa
