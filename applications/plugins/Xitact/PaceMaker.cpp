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
#include "PaceMaker.h"
#include <iostream>
//#include <boost/thread.hpp>


namespace sofa
{

namespace component
{

namespace controller
{

SOFA_DECL_CLASS(PaceMaker)


using namespace sofa::helper::system::thread;

const double PaceMaker::time_scale = 1000 / (double)CTime::getRefTicksPerSec();
double PaceMaker::clockspeed = 0.01;
double PaceMaker::CLOCK = CTime::getRefTime()*time_scale;
HANDLE PaceMaker::clockThread = NULL;

void PaceMaker::runclock()
{
    for(;;)
    {
        CLOCK = sofa::helper::system::thread::CTime::getRefTime()*time_scale;
        Sleep((DWORD)1);

    }
}


void stimulus(void* param)
{
    PaceMaker* myPace = (PaceMaker*)param;//mydata.myPace;

    // Transform frequency in time:
    double time1 = PaceMaker::CLOCK;

    double dt = 0.0;
    double nextTime, time2;

    if (myPace->getFrequency() != 0.0)
        dt = 1/myPace->getFrequency();

    // in ms
    dt = 1000*dt;

    if (!myPace->pToFunc)
    {
        std::cout << "You didn't give a pointer to function to apply." << std::endl;
        return;
    }

    void (*p)(void*) = myPace->pToFunc;
    void* myData = myPace->Pdata;

    for (;;)
    {
        nextTime = time1 + dt;

        time2 = PaceMaker::CLOCK;
        //std::cout <<"nextTime: " << nextTime << " time2: " << time2 << " time1: " << time1 << std::endl;
        /*		while(time2 < nextTime)
        		{
        			time2 = PaceMaker::CLOCK;
        			if (time2 > nextTime)
        				Sleep(0.01);
        		}*/

        if(time2>nextTime)
        {
            time1 = nextTime;
            (*p)(myData);
        }



        /*
        			dt*=0.1;
        			Sleep((DWORD) dt);
        			time1 = nextTime;
        */

    }

}



///default constructor: will create a process runed continually with no end.
PaceMaker::PaceMaker()
    : ProcessID (0)
    , frequency (0.0)
    , endTime (0.0)
    , pToFunc(NULL)
    , Pdata(NULL)
{
    version=0;
}

// Constructor with a given frequency: will run a process with given frequency until no end.
PaceMaker::PaceMaker(double fr)
    : ProcessID (0)
    , frequency (fr)
    , endTime (0.0)
    , pToFunc(NULL)
    , Pdata(NULL)
{
    version=0;
}


// Constructor with a given frequency and end: will run a process with given frequency until given end time.
PaceMaker::PaceMaker(double fr, double end)
    : ProcessID (0)
    , frequency (fr)
    , endTime (end)
    , pToFunc(NULL)
    , Pdata(NULL)
{
    version=0;
}

PaceMaker::~PaceMaker()
{
    std::cerr<<" STOP PaceMaker version "<<version<<std::endl;

    if (handleThread)
    {
        TerminateThread(handleThread, NULL);
    }

    if (clockThread)
    {
        TerminateThread(clockThread, NULL);
    }
}


bool PaceMaker::createPace()
{
    static int numCreatePace=0;
    numCreatePace++;
    version=numCreatePace;

    std::cout<<"createPace on version"<<version<<std::endl;


    //this->mydata.myPace = this;

    clockThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)PaceMaker::runclock, NULL, 0, NULL);

    // TODO: check si besoin de ne pas demarer tout de suite?
    handleThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)stimulus, (LPVOID)this , 0, NULL);

    if (!this->handleThread)
    {
        std::cerr << "Error while creating thread. " << std::endl;
        return false;
    }

    Sleep(100);
    return true;
}








} // namespace controller

} // namepace component

} // namespace sofa

