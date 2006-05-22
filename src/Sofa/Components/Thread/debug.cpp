#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "debug.h"
#include "CTime.h"

namespace Sofa
{

namespace Components
{

namespace Thread
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

void Trace::checkGL(const char *)
{

}

void Trace::checkGL(const char * /*chaine*/, const char * /*ch2*/,int /*d*/)
{

}

void Trace::print(int level, char *chaine)
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
        exit(0);
    }
}

TraceProfile::TraceProfile(char *name, int index, int size)
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
    beginTime = MeasureTime();
}

void TraceProfile::end(int instant)
{
    endTime = MeasureTime();
    times[instant] += (int)(endTime-beginTime);
}

} // namespace Thread

} // namespace Components

} // namespace Sofa
