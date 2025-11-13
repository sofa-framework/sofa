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

#include <sofa/config.h>

#include <sofa/simulation/task/TaskScheduler.h>

// default
#include <thread>
#include <condition_variable>
#include <memory>
#include <map>
#include <string> 
#include <mutex>
#include <atomic>


namespace sofa::simulation
{

class WorkerThread;

class SOFA_SIMULATION_CORE_API DefaultTaskScheduler : public TaskScheduler
{
    enum
    {
        MAX_THREADS = 16,
        STACKSIZE = 64 * 1024 /* 64K */,
    };
            
public:
            
    // interface

    /**
     * Call stop() and start() if not already initialized
     * @param nbThread
     */
    virtual void init(const unsigned int nbThread = 0) final;

    /**
     * Wait and destroy worker threads
     */
    void stop() final;

    WorkerThread* getCurrent();
    unsigned int getThreadCount(void)  const final { return m_threadCount; }
    const char* getCurrentThreadName() override final;
    int getCurrentThreadType() override final;

    // queue task if there is space, and run it otherwise
    bool addTask(Task* task) override final;
    void workUntilDone(Task::Status* status) override final;
    Task::Allocator* getTaskAllocator() override final;

    // factory methods: name, creator function
    static const char* name() { return "_default"; }
            
    static DefaultTaskScheduler* create();

private:
            
    bool isInitialized() const { return m_isInitialized; }
            
    bool isClosing() const { return m_isClosing; }
            
    void	WaitForWorkersToBeReady();
            
    void	wakeUpWorkers();
            
    WorkerThread* getCurrentThread();
            
    WorkerThread* getWorkerThread(const std::thread::id id);

            
    static const std::string _name;

    std::map< std::thread::id, WorkerThread*> _threads;

    std::atomic<const Task::Status*> m_mainTaskStatus;
    void setMainTaskStatus(const Task::Status* mainTaskStatus);
    bool testMainTaskStatus(const Task::Status*);
            
    std::mutex  m_wakeUpMutex;
            
    std::condition_variable m_wakeUpEvent;
            
    DefaultTaskScheduler();
            
    DefaultTaskScheduler(const DefaultTaskScheduler&) = delete;
            
    ~DefaultTaskScheduler() override;

    /**
     * Create worker threads
     * If the number of required threads is 0, the number of threads will be equal to the
     * result of GetHardwareThreadsCount()
     *
     * @param NbThread
     */
    void start(unsigned int NbThread);
            
    bool m_isInitialized;
            
    unsigned m_workerThreadCount;
            
    volatile bool m_workerThreadsIdle;
            
    bool m_isClosing;
            
    unsigned m_threadCount;
            
    friend class WorkerThread;
};

} // namespace sofa::simulation
