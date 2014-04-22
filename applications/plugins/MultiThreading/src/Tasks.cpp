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
* �A multicore tasking engine in some 500 lines of C�
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


#include "Tasks.h"
#include "TaskSchedulerBoost.h"
#include <boost/thread/thread.hpp>

namespace sofa
{

namespace simulation
{


Task::Task(const Task::Status* pStatus) 
: m_Status(pStatus)
, execTime(ctime_t(),ctime_t()), execThreadIndex(-1)
{
}

Task::~Task()
{
}

bool Task::runTask(WorkerThread* thread)
{
    execThreadIndex = thread->getThreadIndex();
    execTime.first = sofa::helper::system::thread::CTime::getFastTime();
    bool res = run(thread);
    execTime.second = sofa::helper::system::thread::CTime::getFastTime();
    return res;
}

const char* Task::getName()
{
    return "Task";
}

Task::Color Task::getColor()
{
    return Color(0.5f,0.5f,0.5f,1.0f);
}

} // namespace simulation

} // namespace sofa
