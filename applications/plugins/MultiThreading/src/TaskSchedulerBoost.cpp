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

    mThread[0] = new WorkerThread( this, 0 );
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


WorkerThread* TaskScheduler::getWorkerThread(const unsigned int index) 
{
    WorkerThread* thread = 0;
    if ( index < mThreadCount ) 
    {
        thread = mThread[index];
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

        // only physical cores. no advantage from hyperthreading.
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
            mThread[iThread] = new WorkerThread(this, iThread);
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



WorkerThread::WorkerThread(TaskScheduler* const& pScheduler, int index)
:mTaskScheduler(pScheduler)
,mThreadIndex(index)
,mStealableTaskCount(0)
,mSpecificTaskCount(0)
,mCurrentStatus(NULL)
,mTaskLogEnabled(false)
,mFinished(false)
{
    assert(pScheduler);
    mTaskMutex.v_ = 0L;
}


WorkerThread::~WorkerThread()
{
    this->mThread;
    //{
    //	release( this->mThread );
    //}
}		

bool WorkerThread::attachToThisThread(TaskScheduler* pScheduler)
{

    mStealableTaskCount		= 0;
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




boost::shared_ptr<boost::thread> WorkerThread::create_and_attach( TaskScheduler* const & taskScheduler)
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

int WorkerThread::getThreadIndex()
{
    return mThreadIndex;
}

void WorkerThread::enableTaskLog(bool val)
{
    mTaskLogEnabled = val;
    if (!val)
    {
        mTaskLog.clear();
    }
}

void WorkerThread::clearTaskLog()
{
    mTaskLog.clear();
}

const std::vector<Task*>& WorkerThread::getTaskLog()
{
    return mTaskLog;
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
        Task*		    pTask;
        Task::Status*	pPrevStatus = NULL;

        while (popTask(&pTask))
        {
            // run
            pPrevStatus = mCurrentStatus;
            mCurrentStatus = pTask->getStatus();
            
            if (mTaskLogEnabled)
                mTaskLog.push_back(pTask);

            pTask->runTask(this);

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
    
    Task* task=NULL;
    unsigned* taskCount=NULL;
    ///< deal with specific task list first.
    if(mSpecificTaskCount > 0)
    {
        taskCount =&mSpecificTaskCount;
        task      =mSpecificTask[*taskCount-1];
    }
    else if(mStealableTaskCount > 0)
    {
        taskCount=&mStealableTaskCount;
        task     =mStealableTask[*taskCount-1];
    }

    if(task == NULL || taskCount==NULL)
    {
        return false;
    }

    // pop from top of the pile
    *outTask = task;
    --*taskCount;
    return true;
}


bool WorkerThread::pushTask(Task* task, Task* taskArray[], unsigned* taskCount )
{
    // if we're single threaded return false
    if ( mTaskScheduler->getThreadCount()<2 ) 
        return false;

    {
        SpinMutexLock lock( &mTaskMutex );

        if (*taskCount >= Max_TasksPerThread )
            return false;
        if( task->getStatus()==NULL ) {
          return false;
        }
        task->getStatus()->MarkBusy(true);
        taskArray[*taskCount] = task;
        ++*taskCount;
    }

    if (!mTaskScheduler->mainTaskStatus)
    {
        mTaskScheduler->mainTaskStatus = task->getStatus();
        mTaskScheduler->wakeUpWorkers();
    }

    return true;
}

bool WorkerThread::addStealableTask(Task* task)
{
    if (pushTask(task,mStealableTask,&mStealableTaskCount))
        return true;
    
    if (mTaskLogEnabled)
        mTaskLog.push_back(task);

    task->runTask(this);

    return false;
}

bool WorkerThread::addSpecificTask(Task* task)
{
    if (pushTask(task,mSpecificTask,&mSpecificTaskCount))
        return true;
    
    if (mTaskLogEnabled)
        mTaskLog.push_back(task);

    task->runTask(this);

    return false;
}

void WorkerThread::runTask(Task* task)
{
    if (mTaskLogEnabled)
        mTaskLog.push_back(task);

    task->runTask(this);
}

bool WorkerThread::giveUpSomeWork(WorkerThread* idleThread)
{
    SpinMutexLock lock;

    if ( !lock.try_lock( &mTaskMutex ) ) 
        return false;

    if (!mStealableTaskCount)
        return false;

    SpinMutexLock	lockIdleThread( &idleThread->mTaskMutex );

    if ( idleThread->mStealableTaskCount )
        return false;

    unsigned int count = (mStealableTaskCount+1) /2;

    Task** p = idleThread->mStealableTask;

    unsigned int iTask;
    for( iTask=0; iTask< count; ++iTask)
    {
        *p++ = mStealableTask[iTask];
        mStealableTask[iTask] = NULL;
    }
    idleThread->mStealableTaskCount = count;

    for( p = mStealableTask; iTask<mStealableTaskCount; ++iTask)
    {
        *p++ = mStealableTask[iTask];
    }
    mStealableTaskCount -= count;

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

        if ( mStealableTaskCount ) 
            return true;
    }

    return false;
}

} // namespace simulation

} // namespace sofa
