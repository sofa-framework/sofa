/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef MultiThreadingTasks_h__
#define MultiThreadingTasks_h__

#include <MultiThreading/config.h>

#include <boost/detail/atomic_count.hpp>
#include <boost/pool/singleton_pool.hpp>

#include <sofa/helper/system/atomic.h>
#include <boost/thread/mutex.hpp>

namespace sofa
{

	namespace simulation
	{

		class WorkerThread;
		class TaskScheduler;


		class SOFA_MULTITHREADING_PLUGIN_API Task
		{
		public:


			// Task Status class definition
			class Status
			{
			public:
				Status();

				bool IsBusy() const;

				

			private:

				void MarkBusy(bool bBusy);

				/*volatile*/ boost::detail::atomic_count mBusy;

				friend class WorkerThread;
			};



		protected:

			Task(const Task::Status* status);

		
		public:
			
			virtual ~Task();

			//struct TaskTag{};
			//typedef boost::singleton_pool<TaskTag, sizeof(*this)> memory_pool;


			virtual bool run(WorkerThread* thread) = 0;


		private:

            Task(const Task& /*task*/) {}
            Task& operator= (const Task& /*task*/) {return *this;}


		protected:

			inline Task::Status* getStatus(void) const;



			const Task::Status*	m_Status;

			friend class WorkerThread;

		};



		// This task is called once by each thread used by the TasScheduler
		// this is useful to initialize the thread specific variables
		class SOFA_MULTITHREADING_PLUGIN_API ThreadSpecificTask : public Task
		{

		public:

			//InitPerThreadDataTask(volatile long* atomicCounter, boost::mutex* mutex, TaskStatus* pStatus );
            ThreadSpecificTask(helper::system::atomic<int>* atomicCounter, boost::mutex* mutex, Task::Status* pStatus );

			virtual ~ThreadSpecificTask();

			virtual bool runThreadSpecific()  {return true;}

			virtual bool runCriticalThreadSpecific() {return true;}

		private:

			virtual bool run(WorkerThread* );

			//volatile long* mAtomicCounter;
			helper::system::atomic<int>* mAtomicCounter;

			boost::mutex*	 mThreadSpecificMutex;

		};




		// not used yet
		template<class T>
		class TaskAllocator
		{
			struct TaskBaseTag{};
			typedef boost::singleton_pool<TaskBaseTag, sizeof(T)> memory_pool; 

		public:
            static inline void* operator new (std::size_t /*size*/)
			{
				return memory_pool::malloc();
			}

			static inline void operator delete (void* ptr) 
			{
				memory_pool::free(ptr);

			}

		};


	} // namespace simulation

} // namespace sofa


#include "Tasks.inl"


#endif // MultiThreadingTasks_h__
