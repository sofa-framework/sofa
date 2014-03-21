/*                               nulstein @ Evoke 2009
*
*
* ____________________________________
* Copyright 2009 Intel Corporation
* All Rights Reserved
*
* Permission is granted to use, copy, distribute and prepare derivative works of this
* software for any purpose and without fee, provided, that the above copyright notice
* and this statement appear in all copies.  Intel makes no representations about the
* suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
* INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
* INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
* INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
* assume any responsibility for any errors which may appear in this software nor any
* responsibility to update it.
* ____________________________________
*
*
* A multicore tasking engine in some 500 lines of C
* This is the code corresponding to the seminar on writing a task-scheduler suitable 
* for use in multicore optimisation of small prods by Jerome Muffat-Meridol.
*
* Credits :
* -=-=-=-=-
*  .music taken from M40-Southbound, by Ghaal (c)2009
*  .liposuction advice from Matt Pietrek
*     http://www.microsoft.com/msj/archive/S572.aspx
*  .ordering display list ideas based on Christer Ericson's article 
*     http://realtimecollisiondetection.net/blog/?p=86
*  .Approximate Math Library by Alex Klimovitski, Intel GmbH
*  .kkrunchy packed this exe, kudos to ryg/farbrausch
*     http://www.farbrausch.de/~fg/kkrunchy/
*/

#ifndef TaskSchedulerBoost_h__
#define TaskSchedulerBoost_h__

#include <MultiThreading/config.h>

#include "Tasks.h"

#include <boost/pool/pool.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/tss.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/mutex.hpp>

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

    WorkerThread(TaskScheduler* const& taskScheduler, int index);

    ~WorkerThread();

    static WorkerThread* getCurrent();

    // queue task if there is space, and run it otherwise
    bool addTask(Task* pTask);	

    void workUntilDone(Task::Status* status);

    Task::Status* getCurrentStatus() const {return mCurrentStatus;}

    boost::detail::spinlock* getTaskMutex() const {return &mTaskMutex;}
    
    int getThreadIndex();
    
    void enableTaskLog(bool val);
    void clearTaskLog();
    const std::vector<Task*>& getTaskLog();


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

    mutable boost::detail::spinlock	  mTaskMutex;
    Task*		                      mTask[Max_TasksPerThread];
    unsigned			              mTaskCount;								
    Task::Status*	                  mCurrentStatus;	
    TaskScheduler*                    mTaskScheduler;    
    boost::shared_ptr<boost::thread>  mThread;
    int                               mThreadIndex;
    bool mTaskLogEnabled;
    std::vector<Task*> mTaskLog;

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

    WorkerThread* getWorkerThread(const unsigned int index);

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
    volatile unsigned mTargetWorkerCount;	
    volatile unsigned mActiveWorkerCount;
    
    bool              mWorkersIdle;
    bool              mIsClosing;
    unsigned          mThreadCount;



    friend class WorkerThread;
};		

SOFA_MULTITHREADING_PLUGIN_API bool runThreadSpecificTask(WorkerThread* pThread, const Task *pTask );

SOFA_MULTITHREADING_PLUGIN_API bool runThreadSpecificTask(const Task *pTask );

} // namespace simulation

} // namespace sofa


#endif // TaskSchedulerBoost_h__
