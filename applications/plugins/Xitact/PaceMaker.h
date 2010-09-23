#ifndef SOFA_COMPONENT_PACEMAKER_H
#define SOFA_COMPONENT_PACEMAKER_H

#include <iostream>
#include <sofa/helper/system/thread/CTime.h>
#include "initXitact.h"
namespace sofa
{

namespace component
{

namespace controller
{
class PaceMaker;

/*
struct dataPace
{
	PaceMaker* myPace;
	double test;

	dataPace(PaceMaker * puls = NULL, double t = 100)
	{
		test = t;
		myPace = puls;
	}

};*/

class SOFA_XITACTPLUGIN_API PaceMaker
{
public:
    static void runclock();
    static double CLOCK;
    static double clockspeed;
    static const double time_scale;
    static HANDLE clockThread;


    PaceMaker(); //without frequency. Warning function will be called continually
    PaceMaker(double fr);
    PaceMaker(double fr, double end);
    ~PaceMaker();

    //void stimulus();

    bool createPace();

    void functionToApply()
    {
        if (!pToFunc || !Pdata)
            std::cout << "You didn't give a pointer to function to apply. Or a pointer to arguments structure." << std::endl;
        else
            (*pToFunc)(Pdata); //apply function
    };

//	void setFunctionToApply( void (*pToF)()) {this->pToFunc = pToF;}
//	void (*pToFunc)() getFunctionToApply () {return this->pToFunc}


    const int getPaceMakerID() {return ProcessID;};
    const double getFrequency() {return frequency;};
    const double getendTime() {return endTime;};

    //dataPace mydata;
    void (*pToFunc)(void*);
    void * Pdata;

protected:

    int ProcessID;
    double frequency;
    double endTime;
    HANDLE handleThread;

    int version;


};




} // namespace controller

} // namepace component

} // namespace sofa

#endif
