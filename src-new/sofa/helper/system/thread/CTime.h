#ifndef SOFA_HELPER_SYSTEM_THREAD_CTIME_H
#define SOFA_HELPER_SYSTEM_THREAD_CTIME_H

#include <time.h>

#ifdef WIN32
# include <windows.h>
#endif


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

#ifdef WIN32
typedef ULONGLONG ctime_t;
#else
typedef unsigned long long ctime_t;
#endif

class CTime
{
public:
    // Get current reference time
    static volatile ctime_t getRefTime();

    // Get the frequency of the reference timer
    static ctime_t getRefTicksPerSec();

    // Get current time using the fastest available method
    static volatile ctime_t getFastTime();

    // Get the frequency of the fast timer
    static ctime_t getTicksPerSec();

    // Same as getFastTime, but with the additionnal guaranty that it will never decrease.
    static volatile ctime_t getTime();

    // Sleep for the given duration in second
    static void sleep(double s);

protected:
    static ctime_t computeTicksPerSec();
};

} // namespace thread

} // namespace system

} // namespace helper

} // namespace sofa

#endif
