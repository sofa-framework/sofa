#include "TaskScheduler.h"

//#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

	namespace simulation
	{
        
        //thread_local WorkerThreadLockFree* TaskSchedulerLockFree::_workerThreadIndex = nullptr;
		std::map< std::thread::id, WorkerThread*> TaskScheduler::_threads;

        
		TaskScheduler& TaskScheduler::getInstance()
		{
			static TaskScheduler instance;
			return instance;
		}

		TaskScheduler::TaskScheduler()
		{
			_isInitialized = false;
			_threadCount = 0;
			_isClosing = false;

			TaskScheduler::_threads[std::this_thread::get_id()] = new WorkerThread(this);
		}

		TaskScheduler::~TaskScheduler()
		{
			if ( _isInitialized ) 
			{
				stop();
			}
		}

		unsigned TaskScheduler::GetHardwareThreadsCount()
		{
			return std::thread::hardware_concurrency();
		}


		const WorkerThread* TaskScheduler::getWorkerThread(const std::thread::id id)
		{
			auto thread =_threads.find(id);
			if (thread == _threads.end() )
			{
				return nullptr;
			}
			return thread->second;
		}

        void TaskScheduler::init(const unsigned int NbThread )
        {
            if ( _isInitialized )
            {
                if ( NbThread == _threadCount )
                {
                    return;
                }
                stop();
            }
            
            start(NbThread);
        }
        
		void TaskScheduler::start(const unsigned int NbThread )
		{
			stop();

            _isClosing = false;
            _workerThreadsIdle = false;
            _mainTaskStatus	= nullptr;
            
            // only physicsal cores. no advantage from hyperthreading.
            _threadCount = GetHardwareThreadsCount() / 2;
            
            if ( NbThread > 0 && NbThread <= MAX_THREADS  )
            {
                _threadCount = NbThread;
            }

            /* start worker threads */
            for( unsigned int i=1; i<_threadCount; ++i)
            {
				WorkerThread* thread = new WorkerThread(this);
				thread->create_and_attach(this);
				_threads[thread->getId()] = thread;
				thread->start(this);
            }
            
            _workerThreadCount = _threadCount;
            _isInitialized = true;
            return;
		}



		void TaskScheduler::stop()
		{
			_isClosing = true;

			if ( _isInitialized ) 
			{
				// wait for all
				WaitForWorkersToBeReady();
				wakeUpWorkers();
                _isInitialized = false;
                
				for (auto it : _threads)
				{
					// if this is the main thread continue
					if (std::this_thread::get_id() == it.first)
					{
						continue;
					}

					// cpu busy wait
					while (!it.second->isFinished())
					{
						std::this_thread::yield();
					}

					// free memory
					// cpu busy wait: thread.joint call
					delete it.second;
					it.second = nullptr;
				}

                _threadCount = 1;
				_workerThreadCount = 1;

				auto mainThreadIt = _threads.find(std::this_thread::get_id());
				WorkerThread* mainThread = mainThreadIt->second;
				_threads.clear();
				_threads[std::this_thread::get_id()] = mainThread;
			}

			return;
		}

		void TaskScheduler::wakeUpWorkers()
		{
			_workerThreadsIdle = false;
			_wakeUpEvent.notify_all();
		}

		void TaskScheduler::WaitForWorkersToBeReady()
		{
			_workerThreadsIdle = true;
		}


		unsigned TaskScheduler::size()	const
		{
			return _workerThreadCount;
		}



		WorkerThread::WorkerThread(TaskScheduler* const& pScheduler)
        : _tasks()
        , _taskScheduler(pScheduler)
		{
			assert(pScheduler);
			_finished		= false;
            _currentStatus = nullptr;
		}


		WorkerThread::~WorkerThread()
		{
            if (_stdThread.joinable())
            {
                _stdThread.join();
            }
		}

        bool WorkerThread::isFinished()
        {
            return _finished;
        }

		bool WorkerThread::start(TaskScheduler* const& taskScheduler)
		{
			assert(taskScheduler);
			_taskScheduler = taskScheduler;
			_currentStatus = nullptr;

			return  true;
		}

        std::thread* WorkerThread::create_and_attach( TaskScheduler* const & taskScheduler)
        {
            _stdThread = std::thread(std::bind(&WorkerThread::run, this));
            return &_stdThread;
        }

		WorkerThread* WorkerThread::getCurrent()
		{
			auto thread = TaskScheduler::_threads.find(std::this_thread::get_id());
			if (thread == TaskScheduler::_threads.end())
			{
				return nullptr;
			}
			return thread->second;
		}


		void WorkerThread::run(void)
		{
            
			// main loop
            while ( !_taskScheduler->isClosing() )
			{
				Idle();
                
                while ( _taskScheduler->_mainTaskStatus != nullptr)
				{
				
					doWork(0);

				
					if (_taskScheduler->isClosing() )
                    {
                        break;
                    }
				}
			}

			_finished = true;
			return;
		}

        const std::thread::id WorkerThread::getId()
        {
            return _stdThread.get_id();
        }

        void WorkerThread::Idle()
        {
            {
                std::unique_lock<std::mutex> lock( _taskScheduler->_wakeUpMutex );
                // cpu free wait
                _taskScheduler->_wakeUpEvent.wait(lock);
            }
            return;
        }

		void WorkerThread::doWork(Task::Status* status)
		{

			do
			{
				Task*		pTask;
				Task::Status*	pPrevStatus = nullptr;

				while (popTask(&pTask))
				{
					// run
					pPrevStatus = _currentStatus;
					_currentStatus = pTask->getStatus();
				
					pTask->run(this);
                    pTask->~Task();
//                    free(pTask);
					
					_currentStatus->markBusy(false);
					_currentStatus = pPrevStatus;
					
					if ( status && !status->isBusy() )
						return;
				}

				/* check if main work is finished */ 
				if (_taskScheduler->_mainTaskStatus == nullptr)
					return;

			} while (stealTasks());	

		
			return;

		}


		void WorkerThread::workUntilDone(Task::Status* status)
		{
			while (status->isBusy())
			{
				doWork(status);
			}

			if (_taskScheduler->_mainTaskStatus == status)
			{
				_taskScheduler->_mainTaskStatus = nullptr;
			}
		}


		bool WorkerThread::popTask(Task** task)
		{
//            if (!_tasks.pop(task) )
            ScopedLock lock( _taskMutex );
            if (!_tasks.empty() )
            {
                *task = _tasks.back();
                _tasks.pop_back();
                return true;
            }
            *task = nullptr;
            return false;
		}


		bool WorkerThread::pushTask(Task* task)
		{
            // if we're single threaded return false
            if ( _taskScheduler->getThreadCount()<2 )
            {
                return false;
            }
            
            ScopedLock lock( _taskMutex );
             task->getStatus()->markBusy(true);
            _tasks.push_back(task);
            
            if (!_taskScheduler->_mainTaskStatus)
            {
                _taskScheduler->_mainTaskStatus = task->getStatus();
                _taskScheduler->wakeUpWorkers();
            }
            
            return true;
		}

		bool WorkerThread::addTask(Task* task)
		{
			if (pushTask(task))
            {
                return true;
            }
			
			task->run(this);
			return false;
		}


		bool WorkerThread::giveUpSomeWork(WorkerThread* idleThread)
		{
            ScopedLock lock( _taskMutex );
            Task* stealedTask = nullptr;
            if (!_tasks.empty() )
            {
                stealedTask = _tasks.front();
                _tasks.pop_front();
                idleThread->_tasks.push_back(stealedTask);
                return true;
            }
            return false;
		}


		bool WorkerThread::stealTasks()
		{

			for (auto it : TaskScheduler::_threads)
			{
				// if this is the main thread continue
				if (std::this_thread::get_id() == it.first)
				{
					continue;
				}

				if (it.second->giveUpSomeWork(this))
				{
					return true;
				}
			}

			return false;
		}


		
		// called once by each thread used
		// by the TaskScheduler
		bool runThreadSpecificTask(WorkerThread* thread, const Task * /*task*/ )
		{
            std::atomic<int> atomicCounter;
            atomicCounter = TaskScheduler::getInstance().size();
            
            std::mutex  InitThreadSpecificMutex;
            
            Task::Status status;
            
            const int nbThread = TaskScheduler::getInstance().size();
            
            for (int i=0; i<nbThread; ++i)
            {
                thread->addTask( new ThreadSpecificTaskLockFree( &atomicCounter, &InitThreadSpecificMutex, &status ) );
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
