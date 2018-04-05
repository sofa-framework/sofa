/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_SYSTEM_THREAD_DEBUG_H
#define SOFA_HELPER_SYSTEM_THREAD_DEBUG_H

#include <vector>

#include <sofa/helper/system/thread/CTime.h>
#include <string.h>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

enum SOFA_HELPER_API TraceLevel
{
    TRACE_DEBUG   = 0,
    TRACE_INFO    = 1,
    TRACE_ERROR   = 2,
    TRACE_WARNING = 3,
};

class SOFA_HELPER_API Trace
{
    static int mTraceLevel;
    static int mNbInstance;
public:
    Trace();

    static void setTraceLevel(int level);
    static void print(int level, const char *chaine);
};


class SOFA_HELPER_API TraceProfile
{
public:
    int index;
    char *name;
    int size;
    int *times;
    int sum;

    ctime_t beginTime;
    ctime_t endTime;

    TraceProfile(const char *name, int index, int size);
    ~TraceProfile();

    void addTime(int instant, int time);

    void begin();
    void end(int instant);
};



#ifdef TRACE_ENABLE

#define TRACE_LEVEl(level) { Trace::setTraceLevel(level); }
#define TRACE(level, chaine){ Trace::print((level), (char*)(chaine)); }

#else


#define TRACE_LEVEl(level) { }
#define TRACE(level, chaine){ }

#endif
} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

#endif
