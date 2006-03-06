#ifndef SOFA_COMPONENTS_THREAD_DEBUG_CTIME_H
#define SOFA_COMPONENTS_THREAD_DEBUG_CTIME_H

#include <time.h>

#ifdef WIN32
# include <windows.h>
#else
# include <unistd.h>
# include <sys/time.h>
#endif


/*******************
 * Time measurment *
 *******************/
#ifndef CLOCKS_PER_SEC
# define CLOCKS_PER_SEC 1000000L
extern long clock();
#endif


namespace Sofa
{

namespace Components
{

namespace Thread
{

#ifdef WIN32
typedef LONGLONG ctime_t;

inline ctime_t MeasureTime()
{
    LARGE_INTEGER a;
    QueryPerformanceCounter(&a);
    return(a.QuadPart);
}

inline ctime_t TicksPerSec()
{
    LARGE_INTEGER b;

    QueryPerformanceFrequency(&b);
    return(b.QuadPart);
}

inline void MySleep(float a)
{
    _sleep((long)(1000.0*a));
}

#else /* WIN32 */

/* ***************************************************************************
 *  getusec: Donne le nombre de micro seconde depuis le 1er janvier 1970
 *
 ******************************************************************************/

inline void getusec(unsigned long long &utime)
{
    struct timeval tv;
    gettimeofday(&tv,0);
    utime=tv.tv_sec*1000000;
    utime+=tv.tv_usec;
}

inline unsigned long long getusec()
{
    unsigned long long t;
    getusec(t);
    return t;
}


#if defined(__GNUC__) && defined(RDTSC)

/********************************************************************************
 * gettick: Donne de nombre de tick depuis le demarrage du systeme
 *          ATTENTION: Depend de la frequence du processeur
 *******************************************************************************/

inline void gettick(unsigned long long int &tick)
{
    __asm__ volatile ("RDTSC" : "=A" (tick) );
}

inline unsigned long long int  gettick()
{
    unsigned long long int x;
    gettick(x);
    return x;
}


/*********************************************************************
 * getfrequency: Evalue la frequence d'horloge interne du CPU
 *
 **********************************************************************/

inline void getfrequency(double &cpu_frequency)
{
    unsigned long long tick1,tick2,tick3,tick4, time1, time2;

    gettick(tick1);
    getusec(time1);
    gettick(tick2);

    //  cout << "Timing .";
    for(int unsigned i=0; i<10; i++)
    {
        unsigned long long time1,time2;
        getusec(time1);
        do
        {
            getusec(time2);
        }
        while ( time2-time1<10000);
        //    cout << ".";
    }
    //  cout << "Done. ";

    gettick(tick3);
    getusec(time2);
    gettick(tick4);

    double cpu_frequency_min =(tick3-tick2) * 1000000.0 /(time2 - time1);
    double cpu_frequency_max =(tick4-tick1) * 1000000.0 /(time2 - time1);
    //cout << " Frequency in [" << cpu_frequency_min << " .. " << cpu_frequency_max << "]" << endl;
    cpu_frequency= (cpu_frequency_min + cpu_frequency_max) / 2.0;
}

inline double getfrequency()
{
    double freq;
    getfrequency(freq);
    return freq;
}

/***************************************************************************************************/

typedef unsigned long long ctime_t;

inline ctime_t TicksPerSec()
{
    static double freq=getfrequency();
    return (ctime_t) freq;
}

inline ctime_t MeasureTime()
{
    return gettick();
}


#else /* __GNUC__ && RDTSC */

typedef unsigned long long ctime_t;

inline ctime_t MeasureTime()
{
//  return(clock());
    return getusec();
}
inline ctime_t TicksPerSec()
{
//  return(CLOCKS_PER_SEC);
    return 1000000;
}


#endif /* __GNUC__ && RDTSC */

inline void MySleep(float a)
{
    usleep((unsigned int)(a*1000000.0F));
}

#endif /* WIN32 */

} // namespace Thread

} // namespace Components

} // namespace Sofa

#endif
