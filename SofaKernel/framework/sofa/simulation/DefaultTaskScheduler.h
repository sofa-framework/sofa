/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef DefaultTaskScheduler_h__
#define DefaultTaskScheduler_h__

#include <sofa/config.h>
#include <sofa/helper/system/config.h>

#include <sofa/simulation/TaskScheduler.h>

#include <atomic>

// default
#include <thread>
#include <condition_variable>
#include <memory>
#include <map>
#include <deque>
#include <string> 
#include <mutex>


// workerthread
#include <sofa/simulation/Locks.h>


namespace sofa  {

    namespace simulation
    {

 //#define ENABLE_TASK_SCHEDULER_PROFILER 1     // Comment this line to disable the profiler

#if ENABLE_TASK_SCHEDULER_PROFILER

#include "TaskSchedulerProfiler.h"

#else
        //----------------------
        // Profiler is disabled
        //----------------------
#define DECLARE_TASK_SCHEDULER_PROFILER(name)
#define DEFINE_TASK_SCHEDULER_PROFILER(name)
#define TASK_SCHEDULER_PROFILER(name)

#endif


        class DefaultTaskScheduler;
        class WorkerThread;


        class SOFA_SIMULATION_CORE_API WorkerThread
        {
        public:

            WorkerThread(DefaultTaskScheduler* const& taskScheduler, const int index, const std::string& name = "Worker");

            ~WorkerThread();

            static WorkerThread* getCurrent();

            // queue task if there is space, and run it otherwise
            bool addTask(Task* pTask);

            void workUntilDone(Task::Status* status);

            const Task::Status* getCurrentStatus() const { return _currentStatus; }

            const char* getName() const { return _name.c_str(); }

            int getType() const { return _type; }

            const std::thread::id getId();

            const std::deque<Task*>* getTasksQueue() { return &_tasks; }

            std::uint64_t getTaskCount() { return _tasks.size(); }

            int GetWorkerIndex();

            void* allocate();

            void free(void* ptr);


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

            const std::string _name;

            const int _type;

            simulation::SpinLock _taskMutex;

            std::deque<Task*> _tasks;

            std::thread  _stdThread;

            const Task::Status*	_currentStatus;

            DefaultTaskScheduler*     _taskScheduler;

            // The following members may be accessed by _multiple_ threads at the same time:
            volatile bool	_finished;

            friend class DefaultTaskScheduler;
        };



        class SOFA_SIMULATION_CORE_API DefaultTaskScheduler : public TaskScheduler
        {
            enum
            {
                MAX_THREADS = 16,
                STACKSIZE = 64 * 1024 /* 64K */,
            };

        public:

            // interface

            virtual void init(const unsigned int nbThread = 0) final;
            virtual void stop(void) final;
            virtual unsigned int getThreadCount(void)  const final { return _threadCount; }
            virtual const char* getCurrentThreadName() override final;
            virtual int GetCurrentThreadType() override final;
            
            // queue task if there is space, and run it otherwise
            bool addTask(Task* task) override final;
            void workUntilDone(Task::Status* status) override final;
            Task::Allocator* getTaskAllocator() override final;

        public:

            // factory methods: name, creator function
            static const char* name() { return "_default"; }

            static DefaultTaskScheduler* create();

            static const bool isRegistered;

        private:

            bool isInitialized() { return _isInitialized; }

            bool isClosing(void) const { return _isClosing; }

            void	WaitForWorkersToBeReady();

            void	wakeUpWorkers();

            static unsigned GetHardwareThreadsCount();

            WorkerThread* getCurrentThread();

            const WorkerThread* getWorkerThread(const std::thread::id id);


        private:

            static const std::string _name;

            // TO DO: replace with thread_specific_ptr. clang 3.5 doesn't support C++ 11 thread_local vars on Mac
            //static thread_local WorkerThread* _workerThreadIndex;
            static std::map< std::thread::id, WorkerThread*> _threads;

            const Task::Status*	_mainTaskStatus;

            std::mutex  _wakeUpMutex;

            std::condition_variable _wakeUpEvent;

        private:

            DefaultTaskScheduler();

            DefaultTaskScheduler(const DefaultTaskScheduler&) {}

            ~DefaultTaskScheduler() override;

            void start(unsigned int NbThread);

            bool _isInitialized;

            unsigned _workerThreadCount;

            bool _workerThreadsIdle;

            bool _isClosing;

            unsigned _threadCount;


            friend class WorkerThread;
        };

	} // namespace simulation

} // namespace sofa


#endif // DefaultTaskScheduler_h__
