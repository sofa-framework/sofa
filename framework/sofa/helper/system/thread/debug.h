#ifndef SOFA_HELPER_SYSTEM_THREAD_DEBUG_H
#define SOFA_HELPER_SYSTEM_THREAD_DEBUG_H

#include <vector>

#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace helper
{

namespace system
{

namespace thread
{

enum TraceLevel
{
    TRACE_DEBUG   = 0,
    TRACE_INFO    = 1,
    TRACE_ERROR   = 2,
    TRACE_WARNING = 3,
};

class Trace
{
    static int mTraceLevel;
    static int mNbInstance;
public:
    Trace();

    static void setTraceLevel(int level);
    static void print(int level, char *chaine);
};


class TraceProfile
{
public:
    int index;
    char *name;
    int size;
    int *times;
    int sum;

    ctime_t beginTime;
    ctime_t endTime;

    TraceProfile(char *name, int index, int size);
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
