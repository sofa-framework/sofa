/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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

    virtual const char* getName() const;
    virtual Color getColor() const;
    
    virtual bool runTask(WorkerThread* thread);

    const TimeInterval& getExecTime() const { return execTime; }
    int getExecThreadIndex() const { return execThreadIndex; }
    ctime_t getExecDuration() const { return execTime.second - execTime.first; }

    static bool compareExecDuration(Task* a, Task* b)
    {
        return a->getExecDuration() < b->getExecDuration();
    }
    
    static bool compareExecDurationReverse(Task* a, Task* b)
    {
        return a->getExecDuration() > b->getExecDuration();
    }
protected:
    
    virtual bool run(WorkerThread* thread) = 0;

    Task(const Task::Status* status);

    virtual ~Task();

    inline Task::Status* getStatus(void) const;

    const Task::Status*	m_Status;

    friend class WorkerThread;

    TimeInterval execTime;
    int execThreadIndex;

private:

    //Task(const Task& /*task*/) {}
    //Task& operator= (const Task& /*task*/) {return *this;}
};

} // namespace simulation

} // namespace sofa


#include "Tasks.inl"


#endif // MultiThreadingTasks_h__
