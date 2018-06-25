/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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

#include <atomic>
#include <mutex>

#include <boost/pool/singleton_pool.hpp>


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
                Status() : _busy(0) {}
                
                bool isBusy() const
                {
                    return (_busy.load(std::memory_order_relaxed) > 0);
                }
                
                
                
            private:
                
                void markBusy(bool busy)
                {
                    if (busy)
                    {
                        _busy.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        _busy.fetch_sub(1, std::memory_order_relaxed);
                    }
                }
                
                std::atomic<int> _busy;
                
                friend class WorkerThread;
            };

		protected:

			Task(const Task::Status* status);

		
		public:
			
			virtual ~Task();

			virtual bool run(WorkerThread* thread) = 0;
            
		private:

            Task(const Task& task) {}
            Task& operator= (const Task& task) {return *this;}


		protected:

			inline Task::Status* getStatus(void) const
            {
                return const_cast<Task::Status*>(_status);
            }

			const Task::Status*	_status;

			friend class WorkerThread;

		};



		// This task is called once by each thread used by the TasScheduler
		// this is useful to initialize the thread specific variables
		class SOFA_MULTITHREADING_PLUGIN_API ThreadSpecificTaskLockFree : public Task
		{

		public:

            ThreadSpecificTaskLockFree(std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus );

			virtual ~ThreadSpecificTaskLockFree();

			virtual bool runThreadSpecific()  {return true;}

			virtual bool runCriticalThreadSpecific() {return true;}

		private:

			virtual bool run(WorkerThread* );

			//volatile long* mAtomicCounter;
			std::atomic<int>* _atomicCounter;

			std::mutex*	 _threadSpecificMutex;

		};


	} // namespace simulation

} // namespace sofa


//#include "Tasks.inl"


#endif // MultiThreadingTasksPOC_h__
