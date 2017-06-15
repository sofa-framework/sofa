/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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

#include "TimeoutWatchdog.h"
#include <iostream>

#ifdef SOFA_HAVE_BOOST

namespace sofa
{
namespace helper
{
namespace system
{
namespace thread
{
TimeoutWatchdog::TimeoutWatchdog()
    : timeout_sec(0)
{
}

TimeoutWatchdog::~TimeoutWatchdog()
{
    if(timeout_sec > 0)
    {
        //std::cout << "Waiting for watchdog thread" << std::endl;
        watchdogThread.interrupt();
        watchdogThread.join();
        //std::cout << "Watchdog thread closed" << std::endl;
    }
}

void TimeoutWatchdog::start(unsigned timeout_sec)
{
    this->timeout_sec = timeout_sec;
    if(timeout_sec > 0)
    {
        boost::thread newThread(boost::bind(&TimeoutWatchdog::threadProc, this));
        watchdogThread.swap(newThread);
    }
}

void TimeoutWatchdog::threadProc()
{
    //std::cout << "Entering watchdog thread" << std::endl;

    // sleep method is interruptible, when calling interrupt() from another thread
    // this thread should end inside the sleep method.
    boost::thread::sleep(boost::get_system_time() + boost::posix_time::seconds(timeout_sec));

    if(!boost::this_thread::interruption_requested())
    {
        std::cerr << "The program has been running for more than "
                << timeout_sec <<
                " seconds. It is going to shut down now." << std::endl;
        exit(-1);
    }
}

}
}
}
}

#endif
