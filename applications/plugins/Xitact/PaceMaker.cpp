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
double PaceMaker::clockspeed = 0.1;
double PaceMaker::CLOCK = CTime::getRefTime()*time_scale;
HANDLE PaceMaker::clockThread = NULL;

void PaceMaker::runclock()
{
    for(;;)
    {
        CLOCK = sofa::helper::system::thread::CTime::getRefTime()*time_scale;
        Sleep((DWORD)clockspeed);

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

        Sleep((DWORD)dt);
        time1 = nextTime;
        (*p)(myData);

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
}

// Constructor with a given frequency: will run a process with given frequency until no end.
PaceMaker::PaceMaker(double fr)
    : ProcessID (0)
    , frequency (fr)
    , endTime (0.0)
    , pToFunc(NULL)
    , Pdata(NULL)
{
}


// Constructor with a given frequency and end: will run a process with given frequency until given end time.
PaceMaker::PaceMaker(double fr, double end)
    : ProcessID (0)
    , frequency (fr)
    , endTime (end)
    , pToFunc(NULL)
    , Pdata(NULL)
{
}

PaceMaker::~PaceMaker()
{
    if (!handleThread)
        TerminateThread(handleThread, NULL);

    if (!clockThread)
        TerminateThread(clockThread, NULL);
}


bool PaceMaker::createPace()
{
    //this->mydata.myPace = this;

    clockThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)PaceMaker::runclock, NULL, 0, NULL);

    // TODO: check si besoin de ne pas demarer tout de suite?
    handleThread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)stimulus, (LPVOID)this , 0, NULL);

    if (!this->handleThread)
    {
        std::cerr << "Error while creating thread. " << std::endl;
        return false;
    }

    Sleep(10);
    return true;
}








} // namespace controller

} // namepace component

} // namespace sofa

