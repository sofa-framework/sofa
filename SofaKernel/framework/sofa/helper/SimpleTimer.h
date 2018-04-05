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

template<int nIter=100, int nStep=100>
class TSimpleTimer
{
public:

    enum {T_NSTEPS=nStep, T_NITERS=nIter};

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

    TSimpleTimer()
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
            std::stringstream tmpmsg ;
            tmpmsg << "TIMER after " << timer_niter << " iterations :" << msgendl;
            for (int i=0; i<T_NSTEPS; ++i)
            {
                if (timers_total[i])
                {
                    double tcur = 1000.0 * (double)timers_current[i] / (double) timer_freq;
                    double ttot = 1000.0 * (double)timers_total[i] / (double) (timer_niter * timer_freq);
                    tmpmsg << "  " << i << ". " << timers_name[i] << "\t : " << std::fixed << (tcur < 10 ? "   " : tcur < 100 ? "  " : tcur < 1000 ? " " : "") << tcur << " \tms  ( mean " << (ttot < 10 ? "   " : ttot < 100 ? "  " : ttot < 1000 ? " " : "") << ttot << " \tms ) " << msgendl;
                }
            }
            double tcur = 1000.0 * (double)timer_current / (double) timer_freq;
            double ttot = 1000.0 * (double)timer_total / (double) (timer_niter * timer_freq);
            tmpmsg << "** TOTAL *********\t : " << std::fixed << (tcur < 10 ? "   " : tcur < 100 ? "  " : tcur < 1000 ? " " : "") << tcur << " \tms  ( mean " << (ttot < 10 ? "   " : ttot < 100 ? "  " : ttot < 1000 ? " " : "") << ttot << " \tms ) " ;
            msg_info("SimpleTimer") << tmpmsg.str() ;
        }
    }
};

typedef TSimpleTimer<> SimpleTimer;

}

}

#endif
