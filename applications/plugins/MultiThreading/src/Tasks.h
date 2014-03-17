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

#ifndef MultiThreadingTasks_h__
#define MultiThreadingTasks_h__

#include <MultiThreading/config.h>

#include <boost/detail/atomic_count.hpp>
#include <sofa/helper/system/atomic.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/defaulttype/Vec.h>
#include <boost/thread/mutex.hpp>

namespace sofa
{

namespace simulation
{

class WorkerThread;
class TaskScheduler;

class SOFA_MULTITHREADING_PLUGIN_API Task
{
public:
    // Task Status class definition
    class Status
    {
    public:
        Status();

        bool IsBusy() const;

    private:

        void MarkBusy(bool bBusy);

        /*volatile*/ boost::detail::atomic_count mBusy;

        friend class WorkerThread;
    };

    typedef sofa::helper::system::thread::ctime_t ctime_t;
    typedef std::pair<ctime_t,ctime_t> TimeInterval;
    typedef sofa::defaulttype::Vec4f Color;

    virtual const char* getName();
    virtual Color getColor();
    
    bool runTask(WorkerThread* thread);

    const TimeInterval& getExecTime() const { return execTime; }
    int getExecThreadIndex() const { return execThreadIndex; }

protected:
    
    virtual bool run(WorkerThread* thread) = 0;

    Task(const Task::Status* status);

    inline Task::Status* getStatus(void) const;

    const Task::Status*	m_Status;

    friend class WorkerThread;

    TimeInterval execTime;
    int execThreadIndex;

private:

    //Task(const Task& /*task*/) {}
    //Task& operator= (const Task& /*task*/) {return *this;}
};

// This task is called once by each thread used by the TasScheduler
// this is useful to initialize the thread specific variables
class SOFA_MULTITHREADING_PLUGIN_API ThreadSpecificTask : public Task
{

public:

    //InitPerThreadDataTask(volatile long* atomicCounter, boost::mutex* mutex, TaskStatus* pStatus );
    ThreadSpecificTask(helper::system::atomic<int>* atomicCounter, boost::mutex* mutex, Task::Status* pStatus );

    virtual ~ThreadSpecificTask();

    virtual bool runThreadSpecific()  {return true;}

    virtual bool runCriticalThreadSpecific() {return true;}

private:

    virtual bool run(WorkerThread* );

    //volatile long* mAtomicCounter;
    helper::system::atomic<int>* mAtomicCounter;

    boost::mutex*	 mThreadSpecificMutex;

};


} // namespace simulation

} // namespace sofa


#include "Tasks.inl"


#endif // MultiThreadingTasks_h__
