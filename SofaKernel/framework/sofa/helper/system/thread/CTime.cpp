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
#include <sofa/helper/system/thread/CTime.h>

#ifdef WIN32
# include <windows.h>
# if _MSC_VER >= 1400
#  include <intrin.h>
# endif
#elif defined(_XBOX)
#include <xtl.h>
#elif defined(PS3)
#include <sys/sys_time.h>
#include <sys/timer.h>
#else
# include <unistd.h>
# include <sys/time.h>
#endif
#if defined(__ia64__) && (defined(__EDG_VERSION) || defined(__ECC))
#include <ia64intrin.h>
#endif
#include <iostream>


/********************
 * Time measurement *
 ********************/

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{
#if defined(WIN32) || defined(_XBOX)

ctime_t CTime::getRefTime()
{
    LARGE_INTEGER a;
    QueryPerformanceCounter(&a);
    return(a.QuadPart);
}

ctime_t CTime::getRefTicksPerSec()
{
    LARGE_INTEGER b;
    QueryPerformanceFrequency(&b);
    return(b.QuadPart);
}

void CTime::sleep(double a)
{
    _sleep((long)(a*1000.0));
}

#else /* WIN32 */

ctime_t CTime::getRefTime()
{
#ifndef PS3
    struct timeval tv;
    gettimeofday(&tv,0);
    return ((ctime_t)tv.tv_sec)*1000000 + ((ctime_t)tv.tv_usec);
#else
	system_time_t time = sys_time_get_system_time();
	return (ctime_t)time;
#endif
}

ctime_t CTime::getRefTicksPerSec()
{
    return (ctime_t)1000000;
}

void CTime::sleep(double a)
{
    usleep((unsigned int)(a*1000000.0));
}

#endif /* WIN32 */



#ifdef SOFA_RDTSC
#if defined(_MSC_VER)
ctime_t CTime::getFastTime()
{
#if _MSC_VER >= 1400
    return __rdtsc();
#else
    _asm    _emit 0x0F
    _asm    _emit 0x31
#endif
}
#elif defined(__ia64__)
# if defined(__EDG_VERSION) || defined(__ECC)
ctime_t CTime::getFastTime()
{
    return __getReg(_IA64_REG_AR_ITC);
}
# else
ctime_t CTime::getFastTime()
{
    ctime_t t;
    __asm__ __volatile__("mov %0=ar.itc" : "=r"(t) :: "memory");
    return t;
}
# endif
#elif defined(__powerpc__) || defined(__POWERPC__)
ctime_t CTime::getFastTime()
{
    register unsigned long t_u;
    register unsigned long t_l;
    asm volatile ("mftbu %0" : "=r" (t_u) );
    asm volatile ("mftb %0" : "=r" (t_l) );
    return (((ctime_t)t_u)<<32UL) | t_l;
}
#elif defined(__GNUC__)
#if defined(__amd64__)
//#warning 64-bits RDTSC
ctime_t CTime::getFastTime()
{
    unsigned t_u, t_l;
    __asm__ volatile ("rdtsc" : "=a" (t_l), "=d" (t_u) );
    return (((ctime_t)t_u)<<32UL) | t_l;
}
#else
//#warning 32-bits RDTSC
ctime_t CTime::getFastTime()
{
    ctime_t t;
    __asm__ volatile ("rdtsc" : "=A" (t) );

    return t;
}
#endif
#else
#error RDTSC not supported on this platform
#endif

ctime_t CTime::computeTicksPerSec()
{
    ctime_t tick1,tick2,tick3,tick4, time1,time2;
    ctime_t reffreq = getRefTicksPerSec();

    tick1 = getFastTime();
    time1 = getRefTime();
    tick2 = getFastTime();

    sleep(0.1);

    tick3 = getFastTime();
    time2 = getRefTime();
    tick4 = getFastTime();

    double cpu_frequency_min =(tick3-tick2) * ((double)reffreq) / (time2 - time1);
    double cpu_frequency_max =(tick4-tick1) * ((double)reffreq) / (time2 - time1);
    std::cout << " CPU Frequency in [" << ((int)(cpu_frequency_min/1000000))*0.001 << " .. " << ((ctime_t)(cpu_frequency_max/1000000))*0.001 << "] GHz" << std::endl;
    return (ctime_t)((cpu_frequency_min + cpu_frequency_max)/2);
}

ctime_t CTime::getTicksPerSec()
{
    static const ctime_t freq = computeTicksPerSec();
    return freq;
}

#else /* SOFA_RDTSC */

ctime_t CTime::getFastTime()
{
    return getRefTime();
}

ctime_t CTime::getTicksPerSec()
{
    return getRefTicksPerSec();
}

#endif /* SOFA_RDTSC */

ctime_t CTime::getTime()
{
    static ctime_t last = 0;
    ctime_t t = getFastTime();
    if (t <= last)
        t = last;
    else
        last = t;
    return t;
}

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

