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
#include <sofa/simulation/task/WorkerThread.h>
#include <sofa/simulation/task/DefaultTaskScheduler.h>

#include <cassert>
#include <mutex>

#ifdef WIN32
#include <processthreadsapi.h>
#endif

namespace sofa::simulation
{

WorkerThread::WorkerThread(DefaultTaskScheduler *const &taskScheduler, const int index, const std::string &name)
        : m_name(name + std::to_string(index)), m_type(0), m_tasks(), m_taskScheduler(taskScheduler)
{
    assert(taskScheduler);
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

bool WorkerThread::isFinished() const
{
    return m_finished.load(std::memory_order_relaxed);
}

bool WorkerThread::start(DefaultTaskScheduler *const &taskScheduler)
{
    SOFA_UNUSED(taskScheduler);
    assert(taskScheduler);
    m_currentStatus = nullptr;

    return true;
}

std::thread *WorkerThread::create_and_attach(DefaultTaskScheduler *const &taskScheduler)
{
    m_taskScheduler = taskScheduler;
    m_stdThread = std::thread([this] { run(); });
    return &m_stdThread;
}

void WorkerThread::run(void)
{
#ifdef WIN32
    const std::wstring widestr = std::wstring(m_name.begin(), m_name.end());
    HRESULT r = SetThreadDescription(
        GetCurrentThread(),
        widestr.c_str()
        );
#endif

    //workerThreadIndex = this;
    //TaskSchedulerDefault::_threads[std::this_thread::get_id()] = this;

    // main loop
    while (!m_taskScheduler->isClosing())
    {
        Idle();

        while (!m_taskScheduler->testMainTaskStatus(nullptr))
        {

            doWork(nullptr);

            if (m_taskScheduler->isClosing())
            {
                break;
            }
        }
    }

    m_finished.store(true, std::memory_order_relaxed);
}

const std::thread::id WorkerThread::getId() const
{
    return m_stdThread.get_id();
}

void WorkerThread::Idle()
{
    std::unique_lock lock(m_taskScheduler->m_wakeUpMutex);
    m_taskScheduler->m_wakeUpEvent.wait(lock,
        [&] { return !m_taskScheduler->m_workerThreadsIdle; });
}

void WorkerThread::doWork(Task::Status *status)
{
    for (;;)// do
    {
        Task *task;

        while (popTask(&task))
        {
            // run task in the queue
            runTask(task);


            if (status && !status->isBusy())
                return;
        }

        // check if main work is finished
        if (m_taskScheduler->testMainTaskStatus(nullptr))
            return;

        if (!stealTask(&task))
            return;

        // run the stolen task
        runTask(task);

    } //;;while (stealTasks());


}

void WorkerThread::runTask(Task *task)
{
    Task::Status *prevStatus = m_currentStatus;
    m_currentStatus = task->getStatus();

    {
        if (task->run() & Task::MemoryAlloc::Dynamic)
        {
            // pooled memory: call destructor and free
            //task->~Task();
            task->operator delete(task, sizeof(*task));
            //delete task;
        }
    }

    m_currentStatus->setBusy(false);
    m_currentStatus = prevStatus;
}

void WorkerThread::workUntilDone(Task::Status *status)
{
    while (status->isBusy())
    {
        doWork(status);
    }

    if (m_taskScheduler->testMainTaskStatus(status))
    {
        m_taskScheduler->setMainTaskStatus(nullptr);
        m_taskScheduler->m_workerThreadsIdle = true;
    }
}


bool WorkerThread::popTask(Task **task)
{
    simulation::ScopedLock lock(m_taskMutex);
    if (!m_tasks.empty())
    {
        *task = m_tasks.back();
        m_tasks.pop_back();
        return true;
    }
    *task = nullptr;
    return false;
}


bool WorkerThread::pushTask(Task *task)
{
    // if we're single threaded return false
    if (m_taskScheduler->getThreadCount() < 2)
    {
        return false;
    }

    {
        simulation::ScopedLock lock(m_taskMutex);
        const int taskId = task->getStatus()->setBusy(true);
        task->m_id = taskId;
        m_tasks.push_back(task);
    }


    if (m_taskScheduler->testMainTaskStatus(nullptr))
    {
        m_taskScheduler->setMainTaskStatus(task->getStatus());
        m_taskScheduler->wakeUpWorkers();
    }

    return true;
}

bool WorkerThread::addTask(Task *task)
{
    if (pushTask(task))
    {
        return true;
    }

    // we are single thread: run the task
    runTask(task);

    return false;
}

bool WorkerThread::stealTask(Task **task)
{
    for (const auto it : m_taskScheduler->_threads)
    {
        // if this is the main thread continue
        if (std::this_thread::get_id() == it.first)
        {
            continue;
        }

        WorkerThread *otherThread = it.second;
        {
            simulation::ScopedLock lock(otherThread->m_taskMutex);
            if (!otherThread->m_tasks.empty())
            {
                *task = otherThread->m_tasks.front();
                otherThread->m_tasks.pop_front();
                return true;
            }
        }

    }

    return false;
}

} // namespace sofa::simulation
