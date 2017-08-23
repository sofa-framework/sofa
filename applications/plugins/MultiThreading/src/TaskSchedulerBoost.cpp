#include "TaskSchedulerBoost.h"



#include <sofa/helper/system/thread/CTime.h>


namespace sofa
{

	namespace simulation
	{

		boost::thread_specific_ptr<WorkerThread> TaskScheduler::mWorkerThreadIndex;


		TaskScheduler& TaskScheduler::getInstance()
		{
			static TaskScheduler instance;

			return instance;
		}

		TaskScheduler::TaskScheduler()
		{
			mIsInitialized = false;
			mThreadCount = 0;
			mIsClosing = false;
			
			readyForWork = false;

			mThread[0] = new WorkerThread( this );
			mThread[0]->attachToThisThread( this );

		}

		TaskScheduler::~TaskScheduler()
		{
			if ( mIsInitialized ) 
			{
				//stop();
			}
			if ( mThread[0] != 0 )
			{
				//delete mThread[0]; 
			}
		}

		unsigned TaskScheduler::GetHardwareThreadsCount()
		{
			return boost::thread::hardware_concurrency();
		}


		const WorkerThread* TaskScheduler::getWorkerThread(const unsigned int index) 
		{
			const WorkerThread* thread = mThread[index];
			if ( index >= mThreadCount ) 
			{
				return thread = 0;
			}
			return thread;
		}

		bool TaskScheduler::start(const unsigned int NbThread )
		{

			if ( mIsInitialized ) 
			{
				stop();
			}

			//if ( !mIsInitialized ) 
			{
				mIsClosing		= false;
				mWorkersIdle		= false;
				mainTaskStatus	= NULL;

				// only physicsal cores. no advantage from hyperthreading.
				mThreadCount = GetHardwareThreadsCount() / 2;

				if ( NbThread > 0 && NbThread <= MAX_THREADS  )
				{
					mThreadCount = NbThread;
				}			


				//mThread[0] =  new WorkerThread( this ) ;
				//mThread[0]->attachToThisThread( this );

				/* start worker threads */ 
				for( unsigned int iThread=1; iThread<mThreadCount; ++iThread)
				{
					//mThread[iThread] = boost::shared_ptr<WorkerThread>(new WorkerThread(this) );
					mThread[iThread] = new WorkerThread(this);
					mThread[iThread]->create_and_attach( this );
					mThread[iThread]->start( this );
				}

				mWorkerCount = mThreadCount;
				mIsInitialized = true;
				return true;
			}
			//else
			//{
			//	return false;
			//}

		}



		bool TaskScheduler::stop()
		{
			unsigned iThread;

			mIsClosing = true;

			if ( mIsInitialized ) 
			{
				// wait for all
				WaitForWorkersToBeReady();
				wakeUpWorkers();

				for(iThread=1; iThread<mThreadCount; ++iThread)
				{
					while (!mThread[iThread]->mFinished)
					{ 
						//mThread[iThread]->join();
						//WorkerThread::release( mThread[iThread] );
						//mThread[iThread].reset();						
					}
				}
				for(iThread=1; iThread<mThreadCount; ++iThread)
				{
					mThread[iThread] = 0;
				}


				mIsInitialized = false;
				mWorkerCount = 1;
			}

			
			return true;
		}



		void TaskScheduler::wakeUpWorkers()
		{
			
			mWorkersIdle = false;

			{
				boost::lock_guard<boost::mutex> lock(wakeUpMutex);
				readyForWork = true;
			}

			wakeUpEvent.notify_all();

		}


		void TaskScheduler::WaitForWorkersToBeReady()
		{
		
			for(unsigned i=0; i<mThreadCount-1; ++i)
			{}

			mWorkersIdle = true;
		}




		unsigned TaskScheduler::size()	const volatile
		{
			return mWorkerCount;
		}



		WorkerThread::WorkerThread(TaskScheduler* const& pScheduler)
			: mTaskScheduler(pScheduler)
		{
			assert(pScheduler);

			mTaskCount		= 0;
			mFinished		= false;
			mCurrentStatus = NULL;
		}


		WorkerThread::~WorkerThread()
		{
			//{
			//	release( this->mThread );
			//}
		}		

        bool WorkerThread::attachToThisThread(TaskScheduler* /*pScheduler*/)
		{

			mTaskCount		= 0;
			mFinished		= false;			

			TaskScheduler::mWorkerThreadIndex.reset( this );

			return true;
		}



		bool WorkerThread::start(TaskScheduler* const& taskScheduler)
		{
			assert(taskScheduler);
			mTaskScheduler = taskScheduler;
			mCurrentStatus = NULL;

			return  mThread != 0;
		}




		boost::shared_ptr<boost::thread> WorkerThread::create_and_attach( TaskScheduler* const & /*taskScheduler*/)
		{

			//boost::shared_ptr<WorkerThread> worker( new WorkerThread(taskScheduler) );
			//if(worker)
			//{
				mThread = boost::shared_ptr<boost::thread>(new boost::thread(boost::bind(&WorkerThread::run, this)));
			//}
			return mThread;
		}


		bool WorkerThread::release()
		{

			if ( mThread.get() != 0 )
			{
				mThread->join();

				return true;
			}
			
			return false;
		}


		WorkerThread* WorkerThread::getCurrent()
		{
			return TaskScheduler::mWorkerThreadIndex.get();
		}


		void WorkerThread::run(void)
		{

			// Thread Local Storage 
			TaskScheduler::mWorkerThreadIndex.reset( this );

			// main loop 
			for(;;)
			{
				Idle();

				if ( mTaskScheduler->isClosing() ) 
					break;


				while (mTaskScheduler->mainTaskStatus)
				{
				
					doWork(0);

				
					if (mTaskScheduler->isClosing() ) 
						break;
				}

			}

			mFinished = true;
			return;
		}


		boost::thread::id WorkerThread::getId()
		{
			return mThread->get_id();
		}


		void WorkerThread::Idle()
		{

			boost::unique_lock<boost::mutex> lock( mTaskScheduler->wakeUpMutex );

			while(!mTaskScheduler->readyForWork)
			{
				mTaskScheduler->wakeUpEvent.wait(lock);
			}

			return;
		}

		


		void WorkerThread::doWork(Task::Status* status)
		{

			do
			{
				Task*		pTask;
				Task::Status*	pPrevStatus = NULL;

				while (popTask(&pTask))
				{
					// run
					pPrevStatus = mCurrentStatus;
					mCurrentStatus = pTask->getStatus();
				
					pTask->run(this);
					
					mCurrentStatus->MarkBusy(false);
					mCurrentStatus = pPrevStatus;
					
					if ( status && !status->IsBusy() ) 
						return;
				}

				/* check if main work is finished */ 
				if (!mTaskScheduler->mainTaskStatus) 
					return;

			} while (stealTasks());	

		
			return;

		}


		void WorkerThread::workUntilDone(Task::Status* status)
		{
			//PROFILE_SYNC_PREPARE( this );

			while (status->IsBusy())
			{
				//boost::this_thread::yield();
				doWork(status);
			}

			//PROFILE_SYNC_CANCEL( this );

			if (mTaskScheduler->mainTaskStatus == status)
			{	

				mTaskScheduler->mainTaskStatus = NULL;

				boost::lock_guard<boost::mutex> lock(mTaskScheduler->wakeUpMutex);
				mTaskScheduler->readyForWork = false;				
			}
		}


		bool WorkerThread::popTask(Task** outTask)
		{
			SpinMutexLock lock( &mTaskMutex );

			//
			if (!mTaskCount) 
				return false;

			Task* task = mTask[mTaskCount-1];


			// pop from top of the pile
			*outTask = task;
			--mTaskCount;
			return true;
		}


		bool WorkerThread::pushTask(Task* task)
		{
			// if we're single threaded return false
			if ( mTaskScheduler->getThreadCount()<2 ) 
				return false;
			 	
			{	
				SpinMutexLock lock( &mTaskMutex );

				
				if (mTaskCount >= Max_TasksPerThread )
					return false;

				
				task->getStatus()->MarkBusy(true);
				mTask[mTaskCount] = task;
				++mTaskCount;
			}

			
			if (!mTaskScheduler->mainTaskStatus)
			{
				mTaskScheduler->mainTaskStatus = task->getStatus();
				mTaskScheduler->wakeUpWorkers();
			}

			return true;
		}

		bool WorkerThread::addTask(Task* task)
		{
			if (pushTask(task))
				return true;

			
			task->run(this);
			return false;
		}


		bool WorkerThread::giveUpSomeWork(WorkerThread* idleThread)
		{	
			SpinMutexLock lock;

			if ( !lock.try_lock( &mTaskMutex ) ) 
				return false;

	
			if (!mTaskCount)
				return false;

		
			SpinMutexLock	lockIdleThread( &idleThread->mTaskMutex );

			if ( idleThread->mTaskCount )
				return false;



			unsigned int count = (mTaskCount+1) /2;

			
			Task** p = idleThread->mTask;

			unsigned int iTask;
			for( iTask=0; iTask< count; ++iTask)
			{
				*p++ = mTask[iTask];
				mTask[iTask] = NULL;
			}
			idleThread->mTaskCount = count;

			
			for( p = mTask; iTask<mTaskCount; ++iTask)
			{
				*p++ = mTask[iTask];
			}
			mTaskCount -= count;

			return true;
		}


		bool WorkerThread::stealTasks()
		{

	
			for( unsigned int iThread=0; iThread<mTaskScheduler->getThreadCount(); ++iThread)
			{
				//WorkerThread*	pThread;

				WorkerThread* pThread = mTaskScheduler->mThread[ (iThread)% mTaskScheduler->getThreadCount() ];
				if ( pThread == this) 
					continue;

				if ( pThread->giveUpSomeWork(this) ) 
					return true;

				if ( mTaskCount ) 
					return true;
			}

			return false;
		}


		
		// called once by each thread used
		// by the TaskScheduler
		bool runThreadSpecificTask(WorkerThread* thread, const Task * /*task*/ )
		{

			
			//volatile long atomicCounter = TaskScheduler::getInstance().size();// mNbThread;
			helper::system::atomic<int> atomicCounter( TaskScheduler::getInstance().size() );

			boost::mutex  InitThreadSpecificMutex;

			Task::Status status;

			const int nbThread = TaskScheduler::getInstance().size();

			for (int i=0; i<nbThread; ++i)
			{
				thread->addTask( new ThreadSpecificTask( &atomicCounter, &InitThreadSpecificMutex, &status ) );
			}


			thread->workUntilDone(&status);

			return true;
		}


		// called once by each thread used
		// by the TaskScheduler
		bool runThreadSpecificTask(const Task *task )
		{
			return runThreadSpecificTask(WorkerThread::getCurrent(), task );
		}




	} // namespace simulation

} // namespace sofa
