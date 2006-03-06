#ifndef SOFA_COMPONENTS_THREAD_DEBUG_H
#define SOFA_COMPONENTS_THREAD_DEBUG_H

#include <vector>

#include "CTime.h"

namespace Sofa
{

namespace Components
{

namespace Thread
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
    static void checkGL(const char *);
    static void checkGL(const char *, const char*, int );
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
#define GLCHECK { }

#endif

} // namespace Thread

} // namespace Components

} // namespace Sofa

#endif
