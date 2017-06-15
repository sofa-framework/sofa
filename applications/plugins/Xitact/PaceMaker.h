/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_PACEMAKER_H
#define SOFA_COMPONENT_PACEMAKER_H

#include <iostream>
#include <sofa/helper/system/thread/CTime.h>
#include <Xitact/config.h>
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
