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

#include "NewOmniDriverEmu.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/helper/Quater.h>
//
////force feedback
#include <SofaHaptics/ForceFeedback.h>
#include <SofaHaptics/NullForceFeedback.h>
//
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/simulation/PauseEvent.h>
//
#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaOpenglVisual/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
//sensable namespace
#include <pthread.h>



double prevTime;

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;
using namespace core::behavior;
using namespace sofa::defaulttype;

NewOmniDriverEmu::NewOmniDriverEmu()
    : forceScale(initData(&forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
    , scale(initData(&scale, 1.0, "scale","Default scale applied to the Phantom Coordinates. "))
    , positionBase(initData(&positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
    , orientationBase(initData(&orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
    , positionTool(initData(&positionTool, Vec3d(0,0,0), "positionTool","Position of the tool in the omni end effector frame"))
    , orientationTool(initData(&orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool in the omni end effector frame"))
    , permanent(initData(&permanent, false, "permanent" , "Apply the force feedback permanently"))
    , omniVisu(initData(&omniVisu, false, "omniVisu", "Visualize the position of the interface in the virtual scene"))
    , simuFreq(initData(&simuFreq, 1000, "simuFreq", "frequency of the \"simulated Omni\""))
    , simulateTranslation(initData(&simulateTranslation, false, "simulateTranslation", "do very naive \"translation simulation\" of omni, with constant orientation <0 0 0 1>"))
    , trajPts(initData(&trajPts, "trajPoints","Trajectory positions"))
    , trajTim(initData(&trajTim, "trajTiming","Trajectory timing"))
    , visu_base(NULL)
    , visu_end(NULL)
{

    this->f_listening.setValue(true);
    data.forceFeedback = new NullForceFeedback();
    noDevice = false;
    moveOmniBase = false;
    executeAsynchro = false;
    omniSimThreadCreated = false;
}

NewOmniDriverEmu::~NewOmniDriverEmu()
{
    if (visu_base)
    {
        delete visu_base;
    }
    if (visu_end)
    {
        delete visu_end;
    }

}

void NewOmniDriverEmu::cleanup()
{
    sout << "NewOmniDriverEmu::cleanup()" << sendl;
}

void NewOmniDriverEmu::init()
{
    std::cout << "[NewOmniEmu] init" << endl;
}

void *hapticSimuExecute( void *ptr )
{

    NewOmniDriverEmu *omniDrv = (NewOmniDriverEmu*)ptr;
    double timeScale = 1.0 / (double)CTime::getTicksPerSec();
    double startTime, endTime, totalTime, realTimePrev = -1.0, realTimeAct;
    double requiredTime = 1.0/double(omniDrv->simuFreq.getValue()) * 1.0/timeScale; // [us]
    double timeCorrection = 0.1 * requiredTime;
    int timeToSleep;

    // construct the "trajectory"
    NewOmniDriverEmu::VecCoord pts = omniDrv->trajPts.getValue();
    unsigned int numPts = pts.size();
    helper::vector<double> tmg = omniDrv->trajTim.getValue();
    unsigned int numSegs = tmg.size();
    double stepTime = 1.0/omniDrv->simuFreq.getValue();

    if (numSegs != (2*numPts - 1))
    {
        std::cerr << "Bad trajectory specification " << std::endl;
        return(0);
    }
    NewOmniDriverEmu::VecCoord stepDiff;
    helper::vector<int> stepNum;

    unsigned int seg = 0;
    for (unsigned int np = 0; np < numPts; np++)
    {
        //for the point
        unsigned int n = tmg[seg]*omniDrv->simuFreq.getValue();
        stepNum.push_back(n);
        cout << "N pts = " << n << endl;
        NewOmniDriverEmu::Coord crd;
        cout << " adding  " << crd << endl;
        stepDiff.push_back(crd);

        //for the line
        if (np < numPts-1)
        {
            seg++;
            n = tmg[seg]*omniDrv->simuFreq.getValue();
            cout << "N lin = " << n << endl;
            stepNum.push_back(n);
            Vec3d dx = (pts[np+1].getCenter() - pts[np].getCenter())/double(n);
            helper::Quater<double> dor;  ///TODO difference for rotations!!!
            NewOmniDriverEmu::Coord crd(dx, dor);
            cout << "adding " << crd << endl;
            stepDiff.push_back(crd);
        }
        seg++;
    }

    std::cout << " stepNum = " << stepNum << std::endl;
    std::cout << " stepDiff = " << stepDiff << std::endl;

    //trajectory done

    std::cout << "TimeScale = " << timeScale << std::endl;

    SolidTypes<double>::SpatialVector temp1, temp2;

    long long unsigned asynchroStep=0;
    double averageFreq = 0.0, minimalFreq=1e10;

    unsigned int actSeg = 0;
    unsigned int actStep = 0;

    sofa::helper::Quater<double> actualRot;
    sofa::defaulttype::Vec3d actualPos = pts[0].getCenter();

    cout << "numSegs = " << numSegs << endl;
    cout << "numSegs = " << numSegs << endl;

    while (true)
    {
        if (omniDrv->executeAsynchro)
        {
            startTime = double(omniDrv->thTimer->getTime());

            //compute the actual position
            if (actSeg < numSegs)
            {
                if (actStep < stepNum[actSeg])
                {
                    actualPos += stepDiff[actSeg].getCenter();
                    //cout << "Adding [" << actStep << "] " << stepDiff[actSeg] << endl;
                    actStep++;
                }
                else
                {
                    actStep=0;
                    actSeg++;
                    //cout << "Changing " << endl;
                }
            }
            //else
            //    cout << "Finished" << endl;


            omniDrv->data.servoDeviceData.pos = actualPos;
            omniDrv->data.servoDeviceData.quat = actualRot;
            SolidTypes<double>::Transform baseOmni_H_endOmni(actualPos * omniDrv->data.scale, actualRot);
            SolidTypes<double>::Transform world_H_virtualTool = omniDrv->data.world_H_baseOmni * baseOmni_H_endOmni * omniDrv->data.endOmni_H_virtualTool;

            omniDrv->data.forceFeedback->computeWrench(world_H_virtualTool,temp1,temp2);

            realTimeAct = double(omniDrv->thTimer->getTime());
            if (asynchroStep > 0)
            {
                double realFreq = 1.0/( (realTimeAct - realTimePrev)*timeScale );
                averageFreq += realFreq;
                //std::cout << "actual frequency = " << realFreq << std::endl;
                if (realFreq < minimalFreq)
                    minimalFreq = realFreq;

                if ( ((asynchroStep+1) % 1000) == 0)
                {
                    std::cout << "Average frequency of the loop = " << averageFreq/double(asynchroStep) << " Hz " << std::endl;
                    std::cout << "Minimal frequency of the loop = " << minimalFreq << " Hz " << std::endl;
                }
            }

            realTimePrev = realTimeAct;
            asynchroStep++;

            endTime = double(omniDrv->thTimer->getTime());  //[s]
            totalTime = (endTime - startTime);  // [us]
            timeToSleep = int( (requiredTime - totalTime) - timeCorrection); //  [us]
            if (timeToSleep > 0)
            {
                usleep(timeToSleep);
                //std::cout << "Frequency OK, computation time: " << totalTime << std::endl;
            }
            else
            {
                std::cout << "Cannot achieve desired frequency, computation too slow: " << totalTime << std::endl;
            }

        }
        else
        {
            //std::cout << "Running Asynchro without action" << std::endl;
            usleep(10000);
        }


    }
}

void NewOmniDriverEmu::bwdInit()
{
    sout<<"NewOmniDriverEmu::bwdInit() is called"<<sendl;
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
    ForceFeedback *ff = context->getTreeObject<ForceFeedback>();

    data.forceFeedback = ff;

    setDataValue();

    if (!omniSimThreadCreated)
    {
        sout << "Not initializing phantom, starting emulating thread..." << sendl;
        pthread_t hapSimuThread;

        if (thTimer == NULL)
            thTimer = new(CTime);

        if ( pthread_create( &hapSimuThread, NULL, hapticSimuExecute, (void*)this) == 0 )
        {
            sout << "Thread created for Omni simulation" << sendl;
            omniSimThreadCreated=true;
        }
    }
    else
        sout << "Emulating thread already running" << sendl;
}


void NewOmniDriverEmu::setDataValue()
{
    data.scale = scale.getValue();
    data.forceScale = forceScale.getValue();
    Quat q = orientationBase.getValue();
    q.normalize();
    orientationBase.setValue(q);
    data.world_H_baseOmni.set( positionBase.getValue(), q		);
    q=orientationTool.getValue();
    q.normalize();
    data.endOmni_H_virtualTool.set(positionTool.getValue(), q);
    data.permanent_feedback = permanent.getValue();
}

void NewOmniDriverEmu::reset()
{
    std::cout<<"NewOmniDriver::reset() is called" <<std::endl;
    this->reinit();
}

void NewOmniDriverEmu::reinitVisual()
{
    cout << "NewOmniDriver::reinitVisual() is called " << endl;
    if(visu_base!=NULL)
    {
        cout << "visu_base = " << visu_base << endl;
        delete(visu_base);
        visu_base = new sofa::component::visualmodel::OglModel();
        visu_base->fileMesh.setValue("mesh/omni_test2.obj");
        visu_base->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_end->setColor(1.0f,1.0f,1.0f,1.0f);
        visu_base->init();
        visu_base->initVisual();
        visu_base->updateVisual();
        visu_base->applyRotation(orientationBase.getValue());
        visu_base->applyTranslation( positionBase.getValue()[0],positionBase.getValue()[1], positionBase.getValue()[2]);

    }

    if (visu_end != NULL)
    {
        //serr<<"create visual model for NewOmniDriver end"<<sendl;
        cout << "visu_end = " << visu_end << endl;
        delete(visu_end);
        visu_end = new sofa::component::visualmodel::OglModel();
        visu_end->fileMesh.setValue("mesh/stylus.obj");
        visu_end->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_end->setColor(1.0f,0.3f,0.0f,1.0f);
        visu_end->init();
        visu_end->initVisual();
        visu_end->updateVisual();
    }


}

void NewOmniDriverEmu::reinit()
{
    std::cout<<"NewOmniDriver::reinit() is called" <<std::endl;
    this->cleanup();
    this->bwdInit();
    this->reinitVisual();
    std::cout<<"NewOmniDriver::reinit() done" <<std::endl;


//////////////// visu_base: place the visual model of the NewOmniDriver


    //sofa::component::visualmodel::RigidMappedModel::VecCoord* x_rigid = visu_base->getRigidX();
    // x_rigid->resize(1);
    //(*x_rigid)[0].getOrientation() = q;
    //(*x_rigid)[0].getCenter() =  positionBase.getValue();
    //double s =
    //this->scale=Vector3(this->)

}

void NewOmniDriverEmu::draw()
{
    //cout << "NewOmniDriver::draw is called" << endl;
    if(omniVisu.getValue())
    {
        if (visu_base == NULL)
        {
            cout << "Creating visu_base" << endl;
            // create visual object
            //serr<<"create visual model for NewOmniDriver base"<<sendl;
            visu_base = new sofa::component::visualmodel::OglModel();
            visu_base->fileMesh.setValue("mesh/omni_test2.obj");
            visu_base->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
            visu_base->init();
            visu_base->initVisual();
            visu_base->updateVisual();
            visu_base->applyRotation(orientationBase.getValue());
            visu_base->applyTranslation( positionBase.getValue()[0],positionBase.getValue()[1], positionBase.getValue()[2]);
            //getContext()->addObject(visu_base);
        }


        if (visu_end == NULL)
        {
            //serr<<"create visual model for NewOmniDriver end"<<sendl;
            visu_end = new sofa::component::visualmodel::OglModel();
            visu_end->fileMesh.setValue("mesh/stylus.obj");
            visu_end->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
            visu_end->setColor(1.0f,0.3f,0.0f,1.0f);
            visu_end->init();
            visu_end->initVisual();
            visu_end->updateVisual();
        }

        // compute position of the endOmni in worldframe
        SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
        SolidTypes<double>::Transform world_H_endOmni = data.world_H_baseOmni * baseOmni_H_endOmni ;


        visu_end->xforms.resize(1);
        (visu_end->xforms)[0].getOrientation() = world_H_endOmni.getOrientation();
        (visu_end->xforms)[0].getCenter() =  world_H_endOmni.getOrigin();

        // draw the 2 visual models
        visu_base->drawVisual();
        visu_end->drawVisual();
    }
}

void NewOmniDriverEmu::copyDeviceDataCallback(OmniData *pUserData)
{
    OmniData *data = pUserData; // static_cast<OmniData*>(pUserData);
    memcpy(&data->deviceData, &data->servoDeviceData, sizeof(DeviceData));
    data->servoDeviceData.nupdates = 0;
    data->servoDeviceData.ready = true;
}

void NewOmniDriverEmu::stopCallback(OmniData *pUserData)
{
    OmniData *data = pUserData; // static_cast<OmniData*>(pUserData);
    data->servoDeviceData.stop = true;
}

void NewOmniDriverEmu::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kpe)
{



}

void NewOmniDriverEmu::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kre)
{

    //omniVisu.setValue(false);

}

void NewOmniDriverEmu::handleEvent(core::objectmodel::Event *event)
{

    //std::cout<<"NewEvent detected !!"<<std::endl;


    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        //getData(); // copy data->servoDeviceData to gDeviceData
        //if (!simulateTranslation.getValue()) {
        copyDeviceDataCallback(&data);
        if (data.deviceData.ready)
        {
            cout << "Data ready, event" << endl;
            data.deviceData.quat.normalize();
            //sout << "driver is working ! " << data->servoDeviceData.transform[12+0] << endl;


            /// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
            SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
            SolidTypes<double>::Transform world_H_virtualTool = data.world_H_baseOmni * baseOmni_H_endOmni * data.endOmni_H_virtualTool;


            // store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
            data.forceFeedback->setReferencePosition(world_H_virtualTool);

            /// TODO : SHOULD INCLUDE VELOCITY !!
            sofa::core::objectmodel::HapticDeviceEvent omniEvent(data.deviceData.id, world_H_virtualTool.getOrigin(), world_H_virtualTool.getOrientation() , data.deviceData.m_buttonState);

            this->getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &omniEvent);

            if (moveOmniBase)
            {
                std::cout<<" new positionBase = "<<positionBase_buf<<std::endl;
                visu_base->applyTranslation(positionBase_buf[0] - positionBase.getValue()[0],
                        positionBase_buf[1] - positionBase.getValue()[1],
                        positionBase_buf[2] - positionBase.getValue()[2]);
                positionBase.setValue(positionBase_buf);
                setDataValue();
                //this->reinitVisual();
            }
            executeAsynchro=true;
        }
        else
            std::cout<<"data not ready"<<std::endl;
        //} else {


    }

    if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    {
        core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
        if (kpe->getKey()=='Z' ||kpe->getKey()=='z' )
        {
            moveOmniBase = !moveOmniBase;
            std::cout<<"key z detected "<<std::endl;
            omniVisu.setValue(moveOmniBase);


            if(moveOmniBase)
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
}

int NewOmniDriverEmuClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< NewOmniDriverEmu >();

SOFA_DECL_CLASS(NewOmniDriverEmu)


} // namespace controller

} // namespace component

} // namespace sofa
