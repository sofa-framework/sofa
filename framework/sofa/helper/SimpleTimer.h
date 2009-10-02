#ifndef SOFA_HELPER_SIMPLETIMER_H
#define SOFA_HELPER_SIMPLETIMER_H

#include <sofa/helper/system/thread/CTime.h>

namespace sofa
{

namespace helper
{

/**
  Very simple timer

  Usage example :

  sofa::helper::SimpleTimer mytimer;

  void myComputationCode() {

    bool timer = true; // should I print performance stats
    if (timer) mytimer.start("mystep1);

    ... // step 1 code

    if (timer) mytimer.step("mystep2");

    ... // step 2 code

    if (timer) mytimer.stop();
  }


 */
class SimpleTimer
{
public:

    enum {T_NSTEPS=100, T_NITERS=100};

    typedef sofa::helper::system::thread::ctime_t ctime_t;
    typedef sofa::helper::system::thread::CTime CTime;

    ctime_t timer_total;
    ctime_t timer_current;
    ctime_t timer_freq;
    ctime_t timer_start;
    ctime_t timers_start;
    const char* timers_name[T_NSTEPS];
    ctime_t timers_total[T_NSTEPS];
    ctime_t timers_current[T_NSTEPS];
    int timer_niter;
    int timer_nstep;
    const char* timer_lastname;

    SimpleTimer()
    {
        timer_total = 0;
        timer_current = 0;
        timer_freq = 1;
        timer_start = 0;
        timers_start = 0;
        timer_niter = 0;
        timer_nstep = 0;
        timer_lastname = "";
    }

    void start(const char* name)
    {
        if (timer_niter == 0)
        {
            timer_total = 0;
            timer_current = 0;
            timer_freq = CTime::getTicksPerSec();
            for (int i=0; i<T_NSTEPS; ++i) timers_name[i] = "";
            for (int i=0; i<T_NSTEPS; ++i) timers_total[i] = 0;
            for (int i=0; i<T_NSTEPS; ++i) timers_current[i] = 0;
        }
        ctime_t t = CTime::getTime();
        timer_start = t;
        timers_start = t;
        timer_nstep = 0;
        timer_lastname = name;
    }

    void step(const char* name = "")
    {
        if (timer_nstep >= T_NSTEPS) return;
        int i = timer_nstep;
        //if (timer_niter == 0)
        {
            timers_name[i] = timer_lastname;
            timer_lastname = name;
        }
        ctime_t t = CTime::getTime();
        timers_current[i] = t - timers_start;
        timers_start = t;
        timers_total[i] += timers_current[i];
        ++timer_nstep;
    }

    void stop()
    {
        step();
        ++timer_niter;
        ctime_t t = CTime::getTime();
        timer_current = t - timer_start;
        timer_total += timer_current;
        timer_start = t;
        timers_start = t;
        if (timer_niter > 0 && (timer_niter % T_NITERS) == 0)
        {
            std::cout << "TIMER after " << timer_niter << " iterations :" << std::endl;
            for (int i=0; i<T_NSTEPS; ++i)
            {
                if (timers_total[i])
                {
                    double tcur = 1000.0 * (double)timers_current[i] / (double) timer_freq;
                    double ttot = 1000.0 * (double)timers_total[i] / (double) (timer_niter * timer_freq);
                    std::cout << "  " << i << ". " << timers_name[i] << "\t : " << std::fixed << (tcur < 10 ? "   " : tcur < 100 ? "  " : tcur < 1000 ? " " : "") << tcur << " \tms  ( mean " << (ttot < 10 ? "   " : ttot < 100 ? "  " : ttot < 1000 ? " " : "") << ttot << " \tms ) " << std::endl;
                }
            }
            {
                double tcur = 1000.0 * (double)timer_current / (double) timer_freq;
                double ttot = 1000.0 * (double)timer_total / (double) (timer_niter * timer_freq);
                std::cout << "** TOTAL *********\t : " << std::fixed << (tcur < 10 ? "   " : tcur < 100 ? "  " : tcur < 1000 ? " " : "") << tcur << " \tms  ( mean " << (ttot < 10 ? "   " : ttot < 100 ? "  " : ttot < 1000 ? " " : "") << ttot << " \tms ) " << std::endl;
            }
        }
    }
};

}

}

#endif
