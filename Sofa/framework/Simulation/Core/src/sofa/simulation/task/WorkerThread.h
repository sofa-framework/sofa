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
#pragma once

#include <sofa/simulation/config.h>

#include <sofa/simulation/task/Task.h>
#include <sofa/simulation/Locks.h>

#include <thread>
#include <deque>
#include <string>

namespace sofa::simulation
{

class DefaultTaskScheduler;
class Task;

class SOFA_SIMULATION_CORE_API WorkerThread
{
public:

    WorkerThread(DefaultTaskScheduler* const& taskScheduler, int index, const std::string& name = "Worker");

    ~WorkerThread();

    // queue task if there is space, and run it otherwise
    bool addTask(Task* pTask);

    void workUntilDone(Task::Status* status);

    const Task::Status* getCurrentStatus() const { return m_currentStatus; }

    const char* getName() const { return m_name.c_str(); }

    int getType() const { return m_type; }

    const std::thread::id getId() const;

    const std::deque<Task*>* getTasksQueue() { return &m_tasks; }

    std::uint64_t getTaskCount() { return m_tasks.size(); }

private:

    bool start(DefaultTaskScheduler* const& taskScheduler);

    std::thread* create_and_attach(DefaultTaskScheduler* const& taskScheduler);

    void runTask(Task* task);

    // queue task if there is space (or do nothing)
    bool pushTask(Task* pTask);

    // pop task from queue
    bool popTask(Task** ppTask);

    // steal and queue some task from another thread
    bool stealTask(Task** task);

    void doWork(Task::Status* status);

    // thread main loop
    void run(void);

    //void	ThreadProc(void);
    void	Idle(void);

    bool isFinished() const;

    enum
    {
        Max_TasksPerThread = 256
    };

    const std::string m_name;

    const int m_type;

    simulation::SpinLock m_taskMutex;

    std::deque<Task*> m_tasks;

    std::thread  m_stdThread;

    Task::Status*	m_currentStatus;

    DefaultTaskScheduler*     m_taskScheduler;

    // The following members may be accessed by _multiple_ threads at the same time:
    std::atomic<bool>	m_finished;

    friend class DefaultTaskScheduler;
};

} // namespace sofa::simulation
