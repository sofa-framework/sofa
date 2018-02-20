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
#include <cstdio>
#include <cstdlib>
#include <string>

#include <sofa/helper/system/thread/debug.h>
#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

int Trace::mTraceLevel = 0;
int Trace::mNbInstance = 0;
Trace mySingletonTrace;


Trace::Trace()
{

    if( mNbInstance != 0 )
        print(TRACE_WARNING, "Multiple instance of a singleton class");

#ifdef TRACE_ENABLE
    //printf("Trace: [Enabled]\n");
#else
    //printf("Trace: [Disabled]\n");
#endif

    mNbInstance++;
}

void Trace::setTraceLevel(int level)
{
    mTraceLevel = level;
}

void Trace::print(int level, const char *chaine)
{
    switch( level )
    {
    case TRACE_DEBUG:
        printf("DEBUG: %s\n",chaine);
        break;

    case TRACE_INFO:
        printf("INFO: %s\n", chaine);
        break;

    case TRACE_WARNING:
        printf("WARNING: %s\n", chaine);
        break;

    case TRACE_ERROR:
        printf("ERROR: %s\n", chaine );
        exit(EXIT_FAILURE);
    }
}

TraceProfile::TraceProfile(const char *name, int index, int size)
{
    this->index = index;
    this->name = new char[strlen(name)+1];
    strcpy( this->name, name);

    this->size = size;
    this->times = new int[size];
    int i;
    for(i = 0; i < size; i++)
        this->times[i] = 0;
}

TraceProfile::~TraceProfile()
{
    delete( name );
}

void TraceProfile::addTime(int instant, int time)
{
    times[instant] += time;
}

void TraceProfile::begin()
{
    beginTime = CTime::getTime();
}

void TraceProfile::end(int instant)
{
    endTime = CTime::getTime();
    times[instant] += (int)(endTime-beginTime);
}

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

