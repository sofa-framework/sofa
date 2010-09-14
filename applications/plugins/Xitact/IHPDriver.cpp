/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include <IHPDriver.h>

#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/objectmodel/XitactEvent.h>
//
////force feedback
#include <sofa/component/controller/ForceFeedback.h>
#include <sofa/component/controller/NullForceFeedback.h>
//
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
//
#include <sofa/simulation/common/Node.h>
#include <cstring>

#include <sofa/component/visualModel/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
//sensable namespace
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>



namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;
using namespace core::behavior;
using namespace sofa::defaulttype;









void UpdateForceFeedBack(void* toolData)
{
    XiToolDataIHP* myData = static_cast<XiToolDataIHP*>(toolData);

    // Compute actual tool state:
    xiTrocarAcquire();
    XiToolState state;

    xiTrocarQueryStates();
    xiTrocarGetState(myData->indexTool, &state);

    Vector3 dir;
    dir[0] = (double)state.trocarDir[0];
    dir[1] = (double)state.trocarDir[1];
    dir[2] = (double)state.trocarDir[2];

    double toolDpth = state.toolDepth;
    double thetaX= asin(dir[2]);
    double cx = cos(thetaX);
    double thetaZ=0;

    if (cx > 0.0001)
    {
        thetaZ = -asin(dir[0]/cx);
    }
    else
    {
//		this->f_printLog.setValue(true);
        std::cerr<<"WARNING can not found the position thetaZ of the interface"<<std::endl;
    }

    // Call LCPForceFeedBack
    sofa::helper::vector<sofa::defaulttype::Vector1 > currentState;
    sofa::helper::vector<sofa::defaulttype::Vector1 > ForceBack;

    currentState.resize(6);
    currentState[0] = thetaX;
    currentState[0] = thetaZ;
    currentState[0] = state.toolRoll;
    currentState[0] = toolDpth * myData->scale;
    currentState[0] = state.opening;
    currentState[0] = state.opening;

    myData->forceFeedback->computeForce(currentState, ForceBack);


    /*
    z = [0 -sin(state[0]) cos(state[0])];
    x= [1 0 0];
    Mx = forces[0]; //couple calculé selon X
    Mz = forces[1]; //couple calculé selon Z
    if (toolDepth > 0 {
        OP = trocarDir*toolDepth ;
        forceX = Mz/dot(cross(z,OP),x) ;  => normalement devrait etre tjrs egal a Mz /
    toolDepth
        forceZ = Mx/dot(cross(x,OP),z) ;  => attention pas tjs egal à Mx /toolDepth car
    x et OP non tjs perpendiculaires
    }
    tipForce = trocarDir*forces[3]+  // translation (signe ?)
    forceX*x+ // force "crée" au bout de la pince par le moment Mz
    forceZ*z; // force "crée" au bout de la pince par le moement Mx
    */


    double forceX, forceZ; //Y?
    Vector3 tipForce;

    Vector3 z;
    z[0] = 0;
    z[1] = -sin(thetaX);
    z[2] = cx;
    Vector3 x;
    x[0] = 1; x[1] = 0; x[2] = 0;
    double Mx = ForceBack[0][0];
    double Mz = ForceBack[1][0];

    Vector3 OP;

    if (toolDpth > 0.0)
    {
        OP = dir * toolDpth;
        forceX = Mz/dot( cross(z,OP), x);
        forceZ = Mx/dot( cross(x,OP), z);
    }

    tipForce = dir*ForceBack[3/*2?*/][0] + x * forceX + z *forceZ;

    XiToolForce_ ff;
    ff.tipForce[0] = (float)(tipForce[0] * myData->forceScale);
    ff.tipForce[1] = (float)(tipForce[1] * myData->forceScale);
    ff.tipForce[2] = (float)(tipForce[2] * myData->forceScale);

    if ( (abs(ff.tipForce[0]) > FFthresholdX) || (abs(ff.tipForce[1]) > FFthresholdY) || (abs(ff.tipForce[2]) > FFthresholdZ) )
    {
        std::cout << "Error: Force FeedBack has reached a safety threshold! See header file IHPDriver.h." << std::endl;
        std::cout << "F_X: " << ff.tipForce[0] << "F_Y: " << ff.tipForce[1] << "F_Z: " << ff.tipForce[2] << std::endl;
        return;
    }
    ff.rollForce = 0.0f;

    xiTrocarSetForce(0, &ff);
    xiTrocarFlushForces();
}


bool isInitialized = false;

int initDevice(XiToolDataIHP& /*data*/)
{
    if (isInitialized) return 0;
    isInitialized = true;

    const char* vendor = getenv("XITACT_VENDOR");
    if (!vendor || !*vendor)
        vendor = "INRIA_Sophia";
    xiSoftwareVendor(vendor);

    return 0;
}


SOFA_DECL_CLASS(IHPDriver)
int IHPDriverClass = core::RegisterObject("Driver and Controller of IHP Xitact Device")
        .add< IHPDriver >();

IHPDriver::IHPDriver()
    : Scale(initData(&Scale, 1.0, "Scale","Default scale applied to the Phantom Coordinates. "))
    , forceScale(initData(&forceScale, 0.0001, "forceScale","Default scale applied to the force feedback. "))
    , permanent(initData(&permanent, false, "permanent" , "Apply the force feedback permanently"))
    , indexTool(initData(&indexTool, (int)0,"toolIndex", "index of the tool to simulate (if more than 1). Index 0 correspond to first tool."))
    , graspThreshold(initData(&graspThreshold, 0.2, "graspThreshold","Threshold value under which grasping will launch an event."))
    , showToolStates(initData(&showToolStates, false, "showToolStates" , "Display states and forces from the tool."))
    , testFF(initData(&testFF, false, "testFF" , "If true will add force when closing handle. As if tool was entering an elastic body."))
{
    myPaceMaker = NULL;
    _mstate = NULL;
    this->f_listening.setValue(true);
    //data.forceFeedback = new NullForceFeedback();
    noDevice = false;
    graspElasticMode = false;
}

IHPDriver::~IHPDriver()
{
    xiTrocarRelease();
    this->deleteCallBack();
//	if (data.forceFeedback)
//		delete data.forceFeedback;

}

void IHPDriver::cleanup()
{
    sout << "IHPDriver::cleanup()" << sendl;

    isInitialized = false;

    if (permanent.getValue())
        this->deleteCallBack();
}

void IHPDriver::setForceFeedback(LCPForceFeedback<defaulttype::Vec1dTypes>* ff)
{
    // the forcefeedback is already set
    if(data.forceFeedback == ff)
    {
        return;
    }

    if(data.forceFeedback)
        delete data.forceFeedback;
    data.forceFeedback = ff;
};

void IHPDriver::bwdInit()
{

    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
    if (dynamic_cast<core::behavior::MechanicalState<Vec1dTypes>*>(context->getMechanicalState()) == NULL)
    {
        this->f_printLog.setValue(true);
        serr<<"ERROR : no MechanicalState<Vec1dTypes> defined... init of IHPDriver faild "<<sendl;
        this->_mstate = NULL;
        return ;
    }
    else
    {
        this->_mstate = dynamic_cast<core::behavior::MechanicalState<Vec1dTypes>*> (context->getMechanicalState());

    }

    LCPForceFeedback<defaulttype::Vec1dTypes> *ff = context->get<LCPForceFeedback<defaulttype::Vec1dTypes>>();

    if(ff)
    {
        this->setForceFeedback(ff);
        std::cout << "setForceFeedback(ff) ok" << std::endl;
    }
    else
        std::cout << " Error FF" << std::endl;


    setDataValue();



    if(initDevice(data)==-1)
    {
        noDevice=true;
        std::cout<<"WARNING NO DEVICE"<<std::endl;
    }
    //std::cerr  << "IHPDriver::init() done" << std::endl;

    xiTrocarAcquire();
    char name[1024];
    char serial[16];
    int nbr = this->indexTool.getValue();
    xiTrocarGetDeviceDescription(nbr, name);
    xiTrocarGetSerialNumber(nbr,serial );
    //std::cout << "Tool: " << nbr << std::endl;
    //std::cout << "name: " << name << std::endl;
    //std::cout << "serial: " << serial << std::endl;
    xiTrocarQueryStates();
    xiTrocarGetState(nbr, &data.restState);
    xiTrocarRelease();

    data.indexTool = nbr;

    if (this->permanent.getValue() )
        this->createCallBack();

}


void IHPDriver::setDataValue()
{

    data.scale = Scale.getValue();
    data.forceScale = forceScale.getValue();
    data.permanent_feedback = permanent.getValue();
    /*
    Quat q = orientationBase.getValue();
    q.normalize();
    orientationBase.setValue(q);
    data.world_H_baseIHP.set( positionBase.getValue(), q		);
    q=orientationTool.getValue();
    q.normalize();
    data.endIHP_H_virtualTool.set(positionTool.getValue(), q);

    */
}

void IHPDriver::reset()
{
    this->reinit();
}

void IHPDriver::reinitVisual()
{


}

void IHPDriver::reinit()
{
    this->cleanup();
    this->bwdInit();

    this->reinitVisual();
    //this->updateForce();

    if (permanent.getValue()) //if checkBox is changed
        this->createCallBack();
    else
        this->deleteCallBack();
}


void IHPDriver::updateForce()
{
    // Quick FF test. Add force when using handle. Like in documentation.
    int tool = indexTool.getValue();
    float graspReferencePoint[3] = { 0.0f, 0.0f, 0.0f };
    float kForceScale = 100.0;
    XiToolForce manualForce = { 0 };

    // Checking either handle is open or not:
    if ( (data.simuState.opening <= 0.1) && (!graspElasticMode)) //Activate
    {
        graspElasticMode = true;
        for (unsigned int i = 0; i < 3; ++i)
            graspReferencePoint[i] = data.simuState.trocarDir[i] * data.simuState.toolDepth;
    }

    if ( (data.simuState.opening > 0.1) && (graspElasticMode)) //Desactivate
    {
        graspElasticMode = false;
        xiTrocarSetForce(tool, &manualForce);
        xiTrocarFlushForces();
    }

    if (graspElasticMode)
    {
        for (unsigned int i = 0; i<3; ++i)
            manualForce.tipForce[i] = (graspReferencePoint[i] - (data.simuState.trocarDir[i] * data.simuState.toolDepth)) * kForceScale;

        if (showToolStates.getValue())
        {
            char toolID[16];
            xiTrocarGetSerialNumber(tool,toolID);
            std::cout << toolID << " => Forces = " << manualForce.tipForce[0] << " | " << manualForce.tipForce[1] << " | " << manualForce.tipForce[2] << std::endl;
        }

        manualForce.rollForce = 1.0f;
        xiTrocarSetForce(tool, &manualForce);
        xiTrocarFlushForces();
    }
}


void IHPDriver::displayState()
{
    // simple function print the current device state to the screen.
    char toolID[16];
    xiTrocarGetSerialNumber(indexTool.getValue(),toolID);
    XiToolState state = data.simuState;
    std::cout << toolID
            << " => X = " << state.trocarDir[0]
            << ", Y = " << state.trocarDir[1]
            << ", Z = " << state.trocarDir[2]
            << ", Ins = " << state.toolDepth
            << ", Roll(rad) = " << state.toolRoll
            << ", Open = " << state.opening << std::endl;
}

void IHPDriver::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {


        // calcul des angles à partir de la direction proposée par l'interface...
        // cos(ThetaX) = cx   sin(ThetaX) = sx  cos(ThetaZ) = cz   sin(ThetaZ) = sz .
        // au repos (si cx=1 et cz=1) on a  Axe y
        // on commence par tourner autour de x   puis autour de z
        //   [cz  -sz   0] [1   0   0 ] [0]   [ -sz*cx]
        //   [sz   cz   0]*[0   cx -sx]*[1] = [ cx*cz ]
        //    0    0    1] [0   sx  cx] [0]   [ sx    ]



        xiTrocarAcquire();
        XiToolState state;

        xiTrocarQueryStates();
        xiTrocarGetState(indexTool.getValue(), &state);

        // saving informations in class structure.
        data.simuState = state;

        Vector3 dir;

        dir[0] = (double)state.trocarDir[0];
        dir[1] = (double)state.trocarDir[1];
        dir[2] = (double)state.trocarDir[2];



        double thetaX= asin(dir[2]);
        double cx = cos(thetaX);
        double thetaZ=0;

        if (cx > 0.0001)
        {
            thetaZ = -asin(dir[0]/cx);
        }
        else
        {
            this->f_printLog.setValue(true);
            serr<<"WARNING can not found the position thetaZ of the interface"<<sendl;
        }


        if (showToolStates.getValue()) // print tool state
            this->displayState();

        if (testFF.getValue()) // try FF when closing handle
            this->updateForce();


        if(_mstate)
        {
            //Assign the state of the device to the rest position of the device.

            if(_mstate->getSize()>5)
            {
                (*_mstate->getX0())[0].x() = thetaX;
                (*_mstate->getX0())[1].x() = thetaZ;
                (*_mstate->getX0())[2].x() = state.toolRoll;
                (*_mstate->getX0())[3].x() = state.toolDepth*Scale.getValue();
                (*_mstate->getX0())[4].x() =state.opening;
                (*_mstate->getX0())[5].x() =state.opening;
            }
            else
            {
                this->f_printLog.setValue(true);
                serr<<"PROBLEM WITH MSTATE SIZE: must be >= 6"<<sendl;
            }

        }

        // Button and grasp handling event
        XiStateFlags stateFlag;
        stateFlag = state.flags - data.restState.flags;
        if (stateFlag == XI_ToolButtonLeft)
            this->leftButtonPushed();
        else if (stateFlag == XI_ToolButtonRight)
            this->rightButtonPushed();

        if (state.opening < graspThreshold.getValue())
            this->graspClosed();

    }
    /*
    	//std::cout<<"NewEvent detected !!"<<std::endl;


    	if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    	{
    		//getData(); // copy data->servoDeviceData to gDeviceData
    		hdScheduleSynchronous(copyDeviceDataCallback, (void *) &data, HD_MIN_SCHEDULER_PRIORITY);
    		if (data.deviceData.ready)
    		{
    			data.deviceData.quat.normalize();
    			//sout << "driver is working ! " << data->servoDeviceData.transform[12+0] << endl;


    			/// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
    			SolidTypes<double>::Transform baseIHP_H_endIHP(data.deviceData.pos*data.scale, data.deviceData.quat);
    			SolidTypes<double>::Transform world_H_virtualTool = data.world_H_baseIHP * baseIHP_H_endIHP * data.endIHP_H_virtualTool;


    			/// TODO : SHOULD INCLUDE VELOCITY !!
    			sofa::core::objectmodel::XitactEvent IHPEvent(data.deviceData.id, world_H_virtualTool.getOrigin(), world_H_virtualTool.getOrientation() , data.deviceData.m_buttonState);

    			this->getContext()->propagateEvent(&IHPEvent);

    			if (moveIHPBase)
    			{
    				std::cout<<" new positionBase = "<<positionBase_buf<<std::endl;
    				visu_base->applyTranslation(positionBase_buf[0] - positionBase.getValue()[0],
    											positionBase_buf[1] - positionBase.getValue()[1],
    											positionBase_buf[2] - positionBase.getValue()[2]);
    				positionBase.setValue(positionBase_buf);
    				setDataValue();
    				//this->reinitVisual();
    			}

    		}
    		else
    			std::cout<<"data not ready"<<std::endl;



    	}

    	if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    	{
    		core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
    		if (kpe->getKey()=='Z' ||kpe->getKey()=='z' )
    		{
    			moveIHPBase = !moveIHPBase;
    			std::cout<<"key z detected "<<std::endl;
    			visu.setValue(moveIHPBase);


    			if(moveIHPBase)
    			{
    				this->cleanup();
    				positionBase_buf = positionBase.getValue();

    			}
    			else
    			{
    				this->reinit();
    			}
    		}

    		if(kpe->getKey()=='K' || kpe->getKey()=='k')
    		{
    			positionBase_buf.x()=0.0;
    			positionBase_buf.y()=0.5;
    			positionBase_buf.z()=2.6;
    		}

    		if(kpe->getKey()=='L' || kpe->getKey()=='l')
    		{
    			positionBase_buf.x()=-0.15;
    			positionBase_buf.y()=1.5;
    			positionBase_buf.z()=2.6;
    		}

    		if(kpe->getKey()=='M' || kpe->getKey()=='m')
    		{
    			positionBase_buf.x()=0.0;
    			positionBase_buf.y()=2.5;
    			positionBase_buf.z()=2.6;
    		}



    	}

    */
}

void IHPDriver::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kpe)
{
    (void)kpe;


}

void IHPDriver::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kre)
{
    (void)kre;
    //OmniVisu.setValue(false);

}





Quat IHPDriver::fromGivenDirection( Vector3& dir,  Vector3& local_dir, Quat old_quat)
{
    local_dir.normalize();
    Vector3 old_dir = old_quat.rotate(local_dir);
    dir.normalize();

    if (dot(dir, old_dir)<1.0)
    {
        Vector3 z = cross(old_dir, dir);
        z.normalize();
        double alpha = acos(dot(old_dir, dir));

        Quat dq, Quater_result;

        dq.axisToQuat(z, alpha);

        Quater_result =  old_quat+dq;

        //std::cout<<"debug - verify fromGivenDirection  dir = "<<dir<<"  Quater_result.rotate(local_dir) = "<<Quater_result.rotate(local_dir)<<std::endl;

        return Quater_result;
    }

    return old_quat;
}


void IHPDriver::createCallBack()
{
    if (myPaceMaker)
        delete myPaceMaker;

    myPaceMaker = new sofa::component::controller::PaceMaker(1000);
    myPaceMaker->pToFunc = &UpdateForceFeedBack;
    myPaceMaker->Pdata = &data;
    myPaceMaker->createPace();

    //This function create a thread calling stateCallBack() at a given frequence
}


void IHPDriver::deleteCallBack()
{
    if (myPaceMaker)
        delete myPaceMaker;
}


void IHPDriver::stateCallBack()
{
    // this function delete thread
}

void IHPDriver::rightButtonPushed()
{
    this->operation = true;
}

void IHPDriver::leftButtonPushed()
{
    this->operation = false;
}

void IHPDriver::graspClosed()
{
    if (operation)//Right pedal operation
    {
        return;
    }
    else //Left pedal operation
        return;
}





} // namespace controller

} // namespace component

} // namespace sofa
