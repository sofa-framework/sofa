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
#ifndef TaskScheduler_std_h__
#define TaskScheduler_std_h__

#include <MultiThreading/config.h>

#include "Tasks.h"
#include "Locks.h"
//#include "LockFreeDeQueue.h"
//#include "TasksAllocator.h"

#include <thread>
#include <memory>
#include <map>
#include <deque>


namespace sofa
{

	namespace simulation
	{

		class TaskScheduler;
		class WorkerThread;
        class TasksAllocators;

        
		class SOFA_MULTITHREADING_PLUGIN_API WorkerThread
		{
		public:

			WorkerThread(TaskScheduler* const& taskScheduler);

			~WorkerThread();

			static WorkerThread* getCurrent();
            
            static WorkerThread* getThread();

			// queue task if there is space, and run it otherwise
			bool addTask(Task* pTask);

			void workUntilDone(Task::Status* status);

			Task::Status* getCurrentStatus() const {return _currentStatus;}

            const std::thread::id getId();
            
            const std::deque<Task*>* getTasksQueue() {return &_tasks;}
            
			std::uint64_t getTaskCount()  {return _tasks.size(); }
            
            int GetWorkerIndex();
            
            void* allocate();
            
            void free(void* ptr);
            
            
		private:

			bool start(TaskScheduler* const& taskScheduler);

			std::thread* create_and_attach( TaskScheduler* const& taskScheduler);

			// queue task if there is space (or do nothing)
			bool pushTask(Task* pTask);

			// pop task from queue
			bool popTask(Task** ppTask);
			
			// steal and queue some task from another thread 
			bool stealTasks();

			// give an idle thread some work
			bool giveUpSomeWork(WorkerThread* pIdleThread);
			
			void doWork(Task::Status* status);

			// boost thread main loop
			void run(void);

			//void	ThreadProc(void);
            void	Idle(void);

            bool isFinished();
            
		private:

			enum 
			{
				Max_TasksPerThread = 256
			};

            SpinLock _taskMutex;
            
            std::deque<Task*> _tasks;
            
            std::thread  _stdThread;
            
			Task::Status*	_currentStatus;

			TaskScheduler*     _taskScheduler;
            
			// The following members may be accessed by _multiple_ threads at the same time:
			volatile bool	_finished;

			friend class TaskScheduler;
		};




		class SOFA_MULTITHREADING_PLUGIN_API TaskScheduler

		{
			enum
			{
				MAX_THREADS = 16,
				STACKSIZE = 64*1024 /* 64K */,
			};

		public:
			
			static TaskScheduler& getInstance();
			
            void init(const unsigned int NbThread = 0);
            
            bool isInitialized() { return _isInitialized; }

//            void start(unsigned int NbThread);
			
			void stop(void);

			bool isClosing(void) const { return _isClosing; }

			unsigned int getThreadCount(void) const { return _threadCount; }

			void	WaitForWorkersToBeReady();

			void	wakeUpWorkers();

			static unsigned GetHardwareThreadsCount();

			unsigned size()	const;

			const WorkerThread* getWorkerThread(const std::thread::id id);
		

		private:
			
            //static thread_local WorkerThreadLockFree* _workerThreadIndex;

			static std::map< std::thread::id, WorkerThread*> _threads;

			Task::Status*	_mainTaskStatus;

			std::mutex  _wakeUpMutex;

			std::condition_variable _wakeUpEvent;

		private:

			TaskScheduler();
			
			TaskScheduler(const TaskScheduler& ) {}

			~TaskScheduler();

            void start(unsigned int NbThread);
            
			bool _isInitialized;
            
			unsigned _workerThreadCount;

			bool _workerThreadsIdle;

			bool _isClosing;

			unsigned _threadCount;

			friend class WorkerThread;
		};		




		SOFA_MULTITHREADING_PLUGIN_API bool runThreadSpecificTask(WorkerThread* pThread, const Task *pTask );

		SOFA_MULTITHREADING_PLUGIN_API bool runThreadSpecificTask(const Task *pTask );


	} // namespace simulation

} // namespace sofa


#endif // TaskScheduler_std_h__
