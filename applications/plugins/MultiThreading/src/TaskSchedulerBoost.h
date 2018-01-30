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
#ifndef TaskSchedulerBoost_h__
#define TaskSchedulerBoost_h__

#include <MultiThreading/config.h>

#include "Tasks.h"

#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>


namespace sofa
{

	namespace simulation
	{


		class TaskScheduler;
		class WorkerThread;



		class SpinMutexLock
		{
		public:

			SpinMutexLock() : mMutex(0)
			{
			}

			SpinMutexLock(boost::detail::spinlock* pMutex, bool bLock = true)
				: mMutex(pMutex)
			{

				if (bLock) 
				{
					mMutex->lock();
				}
			}

			bool try_lock(boost::detail::spinlock* pMutex)
			{
				if (!pMutex->try_lock()) 
				{
					return false;
				}

				mMutex = pMutex;
				return true;
			}

			~SpinMutexLock()
			{
				if (mMutex) 
					mMutex->unlock();
			}

		private:
			boost::detail::spinlock* mMutex;
		};





		class SOFA_MULTITHREADING_PLUGIN_API WorkerThread
		{
		public:

			WorkerThread(TaskScheduler* const& taskScheduler);

			~WorkerThread();

			static WorkerThread* getCurrent();

			// queue task if there is space, and run it otherwise
			bool addTask(Task* pTask);	

			void workUntilDone(Task::Status* status);

			Task::Status* getCurrentStatus() const {return mCurrentStatus;}

			boost::detail::spinlock* getTaskMutex() const {return &mTaskMutex;}


		private:

			bool start(TaskScheduler* const& taskScheduler);


			boost::shared_ptr<boost::thread> create_and_attach( TaskScheduler* const& taskScheduler);
			
            bool release();

			



		
			boost::thread::id getId();

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

			bool attachToThisThread(TaskScheduler* pScheduler);


		
			

		private:

			enum 
			{
				Max_TasksPerThread = 256
			};


		

            mutable boost::detail::spinlock		mTaskMutex {BOOST_DETAIL_SPINLOCK_INIT};
			Task*		mTask[Max_TasksPerThread];
			unsigned			mTaskCount;								
			Task::Status*	mCurrentStatus;	

		
			TaskScheduler*     mTaskScheduler;    
			boost::shared_ptr<boost::thread>  mThread;  

			// The following members may be accessed by _multiple_ threads at the same time:
			volatile bool	mFinished;


			friend class TaskScheduler;

		};




		class SOFA_MULTITHREADING_PLUGIN_API TaskScheduler

		{
			enum
			{
				MAX_THREADS = 16,
				STACKSIZE = 64*1024 /* 64K */ 
			};

		public:
			
			static TaskScheduler& getInstance();
			

			bool start(const unsigned int NbThread = 0);
			
			bool stop(void);

			bool isClosing(void) const { return mIsClosing; }

			unsigned int getThreadCount(void) const { return mThreadCount; }


			void	WaitForWorkersToBeReady();

			void	wakeUpWorkers();

			static unsigned GetHardwareThreadsCount();

			unsigned size()	const volatile;

			const WorkerThread* getWorkerThread(const unsigned int index);
		

		private:
			
			static boost::thread_specific_ptr<WorkerThread>	mWorkerThreadIndex;

			//boost::thread_group mThreads;
			WorkerThread* 	mThread[MAX_THREADS];

			
			// The following members may be accessed by _multiple_ threads at the same time:
			volatile Task::Status*	mainTaskStatus;

			volatile bool readyForWork;

			boost::mutex  wakeUpMutex;

			boost::condition_variable wakeUpEvent;
			//boost::condition_variable sleepEvent;



		private:

			TaskScheduler();
			
			TaskScheduler(const TaskScheduler& ) {} 

			~TaskScheduler();

			bool mIsInitialized;
			// The following members may be accessed by _multiple_ threads at the same time:
			volatile unsigned mWorkerCount;	
//			volatile unsigned mTargetWorkerCount;
//			volatile unsigned mActiveWorkerCount;


			bool						mWorkersIdle;


			bool mIsClosing;

			unsigned					mThreadCount;

		

			friend class WorkerThread;
		};		



		

		SOFA_MULTITHREADING_PLUGIN_API bool runThreadSpecificTask(WorkerThread* pThread, const Task *pTask );

		SOFA_MULTITHREADING_PLUGIN_API bool runThreadSpecificTask(const Task *pTask );





	} // namespace simulation

} // namespace sofa


#endif // TaskSchedulerBoost_h__
