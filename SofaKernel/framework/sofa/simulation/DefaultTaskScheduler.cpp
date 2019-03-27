#include <sofa/simulation/DefaultTaskScheduler.h>

#include <sofa/helper/system/thread/thread_specific_ptr.h>

#include <cassert>


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
            m_isInitialized = false;
            m_threadCount = 0;
            m_isClosing = false;
            
            // init global static thread local var
            workerThreadIndex = new WorkerThread(this, 0, "Main  ");
            _threads[std::this_thread::get_id()] = workerThreadIndex;// new WorkerThread(this, 0, "Main  ");
            
        }
        
        DefaultTaskScheduler::~DefaultTaskScheduler()
        {
            if ( m_isInitialized ) 
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
            if ( m_isInitialized )
            {
                if ( (NbThread == m_threadCount) || (NbThread==0 && m_threadCount==GetHardwareThreadsCount()) )
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
            
            m_isClosing = false;
            m_workerThreadsIdle = true;
            m_mainTaskStatus	= nullptr;          
            
            // default number of thread: only physicsal cores. no advantage from hyperthreading.
            m_threadCount = GetHardwareThreadsCount();
            
            if ( NbThread > 0 )//&& NbThread <= MAX_THREADS  )
            {
                m_threadCount = NbThread;
            }
            
            /* start worker threads */
            for( unsigned int i=1; i<m_threadCount; ++i)
            {
                WorkerThread* thread = new WorkerThread(this, i);
                thread->create_and_attach(this);
                _threads[thread->getId()] = thread;
                thread->start(this);
            }
            
            m_workerThreadCount = m_threadCount;
            m_isInitialized = true;
            return;
        }
        
        
        
        void DefaultTaskScheduler::stop()
        {
            m_isClosing = true;
            
            if ( m_isInitialized ) 
            {
                // wait for all
                WaitForWorkersToBeReady();
                wakeUpWorkers();
                m_isInitialized = false;
                
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
                
                m_threadCount = 1;
                m_workerThreadCount = 1;
                
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
        
        int DefaultTaskScheduler::getCurrentThreadType()
        {
            WorkerThread* thread = WorkerThread::getCurrent();
            return thread->getType();
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
                std::lock_guard<std::mutex> guard(m_wakeUpMutex);
                m_workerThreadsIdle = false;
            }								
            m_wakeUpEvent.notify_all();
        }
        
        void DefaultTaskScheduler::WaitForWorkersToBeReady()
        {
            m_workerThreadsIdle = true;
        }
        
        
        //unsigned TaskSchedulerDefault::size()	const
        //{
        //	return _workerThreadCount;
        //}
        
        
        
        WorkerThread::WorkerThread(DefaultTaskScheduler* const& pScheduler, const int index, const std::string& name)
        : m_type(0)
        , m_name(name + std::to_string(index))
        , m_tasks()
        , m_taskScheduler(pScheduler)
        {
            assert(pScheduler);
            m_finished.store(false, std::memory_order_relaxed);
            m_currentStatus = nullptr;
        }
        
        
        WorkerThread::~WorkerThread()
        {
            if (m_stdThread.joinable())
            {
                m_stdThread.join();
            }
            m_finished.store(true, std::memory_order_relaxed);
        }
        
        bool WorkerThread::isFinished()
        {
            return m_finished.load(std::memory_order_relaxed);;
        }
        
        bool WorkerThread::start(DefaultTaskScheduler* const& taskScheduler)
        {
            assert(taskScheduler);
            m_taskScheduler = taskScheduler;
            m_currentStatus = nullptr;
            
            return  true;
        }
        
        std::thread* WorkerThread::create_and_attach(DefaultTaskScheduler* const & taskScheduler)
        {
            SOFA_UNUSED(taskScheduler);
            m_stdThread = std::thread(std::bind(&WorkerThread::run, this));
            return &m_stdThread;
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
            while ( !m_taskScheduler->isClosing() )
            {
                Idle();
                
                while ( m_taskScheduler->m_mainTaskStatus != nullptr)
                {
                    
                    doWork(nullptr);
                    
                    
                    if (m_taskScheduler->isClosing() )
                    {
                        break;
                    }
                }
            }
            
            m_finished.store(true, std::memory_order_relaxed);
            return;
        }
        
        const std::thread::id WorkerThread::getId()
        {
            return m_stdThread.get_id();
        }
        
        void WorkerThread::Idle()
        {
            {
                std::unique_lock<std::mutex> lock( m_taskScheduler->m_wakeUpMutex );
                //if (!_taskScheduler->_workerThreadsIdle)
                //{
                //	return;
                //}
                // cpu free wait
                m_taskScheduler->m_wakeUpEvent.wait(lock, [&] {return !m_taskScheduler->m_workerThreadsIdle; });
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
                if (m_taskScheduler->m_mainTaskStatus == nullptr)
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
            Task::Status* prevStatus = m_currentStatus;
            m_currentStatus = task->getStatus();
            
            {
                if (task->run() & Task::MemoryAlloc::Dynamic)
                {
                    // pooled memory: call destructor and free
                    //task->~Task();
                    task->operator delete (task, sizeof(*task));
                    //delete task;
                }
            }
            
            m_currentStatus->setBusy(false);
            m_currentStatus = prevStatus;
        }
        
        void WorkerThread::workUntilDone(Task::Status* status)
        {
            while (status->isBusy())
            {
                doWork(status);
            }
            
            if (m_taskScheduler->m_mainTaskStatus == status)
            {
                m_taskScheduler->m_mainTaskStatus = nullptr;
            }
        }
        
        
        bool WorkerThread::popTask(Task** task)
        {
            TASK_SCHEDULER_PROFILER(Pop);
            
            simulation::ScopedLock lock( m_taskMutex );
            if (!m_tasks.empty() )
            {
                *task = m_tasks.back();
                m_tasks.pop_back();
                return true;
            }
            *task = nullptr;
            return false;
        }
        
        
        bool WorkerThread::pushTask(Task* task)
        {
            // if we're single threaded return false
            if ( m_taskScheduler->getThreadCount()<2 )
            {
                return false;
            }
            
            {
                TASK_SCHEDULER_PROFILER(Push);
                
                simulation::ScopedLock lock(m_taskMutex);
                int taskId = task->getStatus()->setBusy(true);
                task->m_id = taskId;
                m_tasks.push_back(task);
            }
            
            
            if (!m_taskScheduler->m_mainTaskStatus)
            {
                m_taskScheduler->m_mainTaskStatus = task->getStatus();
                m_taskScheduler->wakeUpWorkers();
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
                
                for (auto it : m_taskScheduler->_threads)
                {
                    // if this is the main thread continue
                    if (std::this_thread::get_id() == it.first)
                    {
                        continue;
                    }
                    
                    WorkerThread* otherThread = it.second;
                    
                    {
                        TASK_SCHEDULER_PROFILER(Steal);
                        
                        simulation::ScopedLock lock(otherThread->m_taskMutex);
                        if (!otherThread->m_tasks.empty())
                        {
                            *task = otherThread->m_tasks.front();
                            otherThread->m_tasks.pop_front();
                            return true;
                        }
                    }
                    
                }
            }
            
            return false;
        }
        

	} // namespace simulation

} // namespace sofa
