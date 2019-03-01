#include <sofa/simulation/DefaultTaskScheduler.h>

#include <sofa/helper/system/thread/thread_specific_ptr.h>

#include <assert.h>


namespace sofa
{
	namespace simulation
	{
        
        DEFINE_TASK_SCHEDULER_PROFILER(Push);
        DEFINE_TASK_SCHEDULER_PROFILER(Pop);
        DEFINE_TASK_SCHEDULER_PROFILER(Steal);


        class StdTaskAllocator : public Task::Allocator
        {
        public:

            void* allocate(std::size_t sz) final
            {
                return ::operator new(sz);
            }

            void free(void* ptr, std::size_t sz) final
            {
                ::operator delete(ptr);
            }
        };

        static StdTaskAllocator defaultTaskAllocator;



        // mac clang 3.5 doesn't support thread_local vars
        //static  WorkerThread* WorkerThread::_workerThreadIndex = nullptr;
        SOFA_THREAD_SPECIFIC_PTR(WorkerThread, workerThreadIndex);

        std::map< std::thread::id, WorkerThread*> DefaultTaskScheduler::_threads;


        DefaultTaskScheduler* DefaultTaskScheduler::create()
        {
            return new DefaultTaskScheduler();
        }

        DefaultTaskScheduler::DefaultTaskScheduler()
            : TaskScheduler()
		{
			_isInitialized = false;
			_threadCount = 0;
			_isClosing = false;

            // init global static thread local var
            workerThreadIndex = new WorkerThread(this, 0, "Main  ");
            _threads[std::this_thread::get_id()] = workerThreadIndex;// new WorkerThread(this, 0, "Main  ");
           
		}

        DefaultTaskScheduler::~DefaultTaskScheduler()
		{
			if ( _isInitialized ) 
			{
				stop();
			}
		}


		unsigned DefaultTaskScheduler::GetHardwareThreadsCount()
		{
			return std::thread::hardware_concurrency() / 2;
		}


		const WorkerThread* DefaultTaskScheduler::getWorkerThread(const std::thread::id id)
		{
			auto thread =_threads.find(id);
			if (thread == _threads.end() )
			{
				return nullptr;
			}
			return thread->second;
		}

        Task::Allocator* DefaultTaskScheduler::getTaskAllocator()
        {
            return &defaultTaskAllocator;
        }

        void DefaultTaskScheduler::init(const unsigned int NbThread )
        {
            if ( _isInitialized )
            {
                if ( (NbThread == _threadCount) || (NbThread==0 && _threadCount==GetHardwareThreadsCount()) )
                {
                    return;
                }
                stop();
            }
            
            start(NbThread);
        }
        
		void DefaultTaskScheduler::start(const unsigned int NbThread )
		{
			stop();

            _isClosing = false;
            _workerThreadsIdle = true;
            _mainTaskStatus	= nullptr;          

            // default number of thread: only physicsal cores. no advantage from hyperthreading.
            _threadCount = GetHardwareThreadsCount();
            
            if ( NbThread > 0 )//&& NbThread <= MAX_THREADS  )
            {
                _threadCount = NbThread;
            }

            /* start worker threads */
            for( unsigned int i=1; i<_threadCount; ++i)
            {
                WorkerThread* thread = new WorkerThread(this, i);
				thread->create_and_attach(this);
				_threads[thread->getId()] = thread;
				thread->start(this);
            }
            
            _workerThreadCount = _threadCount;
            _isInitialized = true;
            return;
		}



		void DefaultTaskScheduler::stop()
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
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

        const char* DefaultTaskScheduler::getCurrentThreadName()
        {
            WorkerThread* thread = WorkerThread::getCurrent();
            return thread->getName();
        }

        bool DefaultTaskScheduler::addTask(Task* task)
        {
            WorkerThread* thread = WorkerThread::getCurrent();
            return thread->addTask(task);
        }

        void DefaultTaskScheduler::workUntilDone(Task::Status* status)
        {
            WorkerThread* thread = WorkerThread::getCurrent();
            thread->workUntilDone(status);
        }

		void DefaultTaskScheduler::wakeUpWorkers()
		{
			{
				std::lock_guard<std::mutex> guard(_wakeUpMutex);
				_workerThreadsIdle = false;
			}								
			_wakeUpEvent.notify_all();
		}

		void DefaultTaskScheduler::WaitForWorkersToBeReady()
		{
			_workerThreadsIdle = true;
		}


		//unsigned TaskSchedulerDefault::size()	const
		//{
		//	return _workerThreadCount;
		//}



        WorkerThread::WorkerThread(DefaultTaskScheduler* const& pScheduler, const int index, const std::string& name)
            : _name(name + std::to_string(index))
            , _index(index)
            , _tasks()
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
            _finished = true;
		}

        bool WorkerThread::isFinished()
        {
            return _finished;
        }

		bool WorkerThread::start(DefaultTaskScheduler* const& taskScheduler)
		{
			assert(taskScheduler);
			_taskScheduler = taskScheduler;
			_currentStatus = nullptr;

			return  true;
		}

        std::thread* WorkerThread::create_and_attach(DefaultTaskScheduler* const & taskScheduler)
        {
            SOFA_UNUSED(taskScheduler);
            _stdThread = std::thread(std::bind(&WorkerThread::run, this));
            return &_stdThread;
        }

        WorkerThread* WorkerThread::getCurrent()
        {
            //return workerThreadIndex;
            auto thread = DefaultTaskScheduler::_threads.find(std::this_thread::get_id());
            if (thread == DefaultTaskScheduler::_threads.end())
            {
                return nullptr;
            }
            return thread->second;
        }

		void WorkerThread::run(void)
		{
            
            //workerThreadIndex = this;
            //TaskSchedulerDefault::_threads[std::this_thread::get_id()] = this;

			// main loop
            while ( !_taskScheduler->isClosing() )
			{
				Idle();
                
                while ( _taskScheduler->_mainTaskStatus != nullptr)
				{
				
					doWork(nullptr);

				
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
				//if (!_taskScheduler->_workerThreadsIdle)
				//{
				//	return;
				//}
                // cpu free wait
                _taskScheduler->_wakeUpEvent.wait(lock, [&] {return !_taskScheduler->_workerThreadsIdle; });
            }
            return;
        }

		void WorkerThread::doWork(Task::Status* status)
		{

            for (;;)// do
            {
                Task* task;

                while (popTask(&task))
                {
                    // run task in the queue
                    runTask(task);


                    if (status && !status->isBusy())
                        return;
                }

                // check if main work is finished 
                if (_taskScheduler->_mainTaskStatus == nullptr)
                    return;

                if (!stealTask(&task))
                    return;

                // run the stolen task
                runTask(task);

            } //;;while (stealTasks());	

		
			return;

		}

        void WorkerThread::runTask(Task* task)
        {
            Task::Status* prevStatus = _currentStatus;
            _currentStatus = task->getStatus();

            {
                if (task->run() & Task::MemoryAlloc::Dynamic)
                {
                    // pooled memory: call destructor and free
                    //task->~Task();
                    task->operator delete (task, sizeof(*task));
                    //delete task;
                }
            }

            _currentStatus->setBusy(false);
            _currentStatus = prevStatus;
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
            TASK_SCHEDULER_PROFILER(Pop);

            simulation::ScopedLock lock( _taskMutex );
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
            
            {
                TASK_SCHEDULER_PROFILER(Push);

                simulation::ScopedLock lock(_taskMutex);
                int taskId = task->getStatus()->setBusy(true);
                task->_id = taskId;
                _tasks.push_back(task);
            }
            
            
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
			
            // we are single thread: run the task
            runTask(task);

			return false;
		}

        bool WorkerThread::stealTask(Task** task)
        {
            {
                //TASK_SCHEDULER_PROFILER(StealTask);

                for (auto it : _taskScheduler->_threads)
                {
                    // if this is the main thread continue
                    if (std::this_thread::get_id() == it.first)
                    {
                        continue;
                    }

                    WorkerThread* otherThread = it.second;

                    {
                        TASK_SCHEDULER_PROFILER(Steal);

                        simulation::ScopedLock lock(otherThread->_taskMutex);
                        if (!otherThread->_tasks.empty())
                        {
                            *task = otherThread->_tasks.front();
                            otherThread->_tasks.pop_front();
                            return true;
                        }
                    }

                }
            }

            return false;
        }
		

	} // namespace simulation

} // namespace sofa
