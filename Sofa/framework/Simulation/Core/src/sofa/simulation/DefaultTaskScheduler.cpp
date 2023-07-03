/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/DefaultTaskScheduler.h>

#include <sofa/helper/system/thread/thread_specific_ptr.h>
#include <sofa/simulation/WorkerThread.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

namespace sofa::simulation
{

const bool DefaultTaskSchedulerRegistered = MainTaskSchedulerFactory::registerScheduler(
    DefaultTaskScheduler::name(),
    &DefaultTaskScheduler::create);

class StdTaskAllocator : public Task::Allocator
{
public:
            
    void* allocate(std::size_t sz) final
    {
        return ::operator new(sz);
    }
            
    void free(void* ptr, std::size_t sz) final
    {  
        SOFA_UNUSED(sz);
        ::operator delete(ptr);
    }
};

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
    {
        _threads[std::this_thread::get_id()] = new WorkerThread(this, 0, "Main  ");// new WorkerThread(this, 0, "Main  ");
    }
}
        
DefaultTaskScheduler::~DefaultTaskScheduler()
{
    if ( m_isInitialized ) 
    {
        stop();
    }
}
        
WorkerThread* DefaultTaskScheduler::getWorkerThread(const std::thread::id id)
{
    const auto thread =_threads.find(id);
    if (thread == _threads.end() )
    {
        return nullptr;
    }
    return thread->second;
}
        
Task::Allocator* DefaultTaskScheduler::getTaskAllocator()
{
    static StdTaskAllocator defaultTaskAllocator;
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
            
    // default number of thread: only physical cores. no advantage from hyperthreading.
    m_threadCount = GetHardwareThreadsCount();
            
    if ( NbThread > 0 )//&& NbThread <= MAX_THREADS  )
    {
        m_threadCount = NbThread;
    }
            
    /* start worker threads */
    for( unsigned int i=1; i<m_threadCount; ++i)
    {
        WorkerThread* thread = new WorkerThread(this, int(i));
        thread->create_and_attach(this);
        _threads[thread->getId()] = thread;
        thread->start(this);
    }
            
    m_workerThreadCount = m_threadCount;
    m_isInitialized = true;
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
                
        for (auto [threadId, workerThread] : _threads)
        {
            // if this is the main thread continue
            if (std::this_thread::get_id() == threadId)
            {
                continue;
            }

            // cpu busy wait
            while (!workerThread->isFinished())
            {
                std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // free memory
            // cpu busy wait: thread.joint call
            delete workerThread;
            workerThread = nullptr;
        }
                
        m_threadCount = 1;
        m_workerThreadCount = 1;

        const auto mainThreadIt = _threads.find(std::this_thread::get_id());
        WorkerThread* mainThread = mainThreadIt->second;
        _threads.clear();
        _threads[std::this_thread::get_id()] = mainThread;
    }
            
    return;
}

WorkerThread* DefaultTaskScheduler::getCurrent()
{
    return getWorkerThread(std::this_thread::get_id());
}

const char* DefaultTaskScheduler::getCurrentThreadName()
{
    const WorkerThread* thread = getCurrent();
    return thread->getName();
}
        
int DefaultTaskScheduler::getCurrentThreadType()
{
    const WorkerThread* thread = getCurrent();
    return thread->getType();
}
        
bool DefaultTaskScheduler::addTask(Task* task)
{
    WorkerThread* thread = getCurrent();
    return thread->addTask(task);
}
        
void DefaultTaskScheduler::workUntilDone(Task::Status* status)
{
    WorkerThread* thread = getCurrent();
    thread->workUntilDone(status);
}
        
void DefaultTaskScheduler::wakeUpWorkers()
{
    {
        std::lock_guard guard(m_wakeUpMutex);
        m_workerThreadsIdle = false;
    }								
    m_wakeUpEvent.notify_all();
}
        
void DefaultTaskScheduler::WaitForWorkersToBeReady()
{
    m_workerThreadsIdle = true;
}

} // namespace sofa::simulation
