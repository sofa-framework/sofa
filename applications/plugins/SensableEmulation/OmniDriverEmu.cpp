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

#include "OmniDriverEmu.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/helper/Quater.h>

#include <sofa/core/visual/VisualParams.h>


////force feedback
#include <SofaHaptics/ForceFeedback.h>
#include <SofaHaptics/NullForceFeedback.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/simulation/PauseEvent.h>

#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaOpenglVisual/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>


#ifndef WIN32
#  include <pthread.h>
#else
#  include <boost/thread/thread.hpp>
#endif


#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>

#ifdef WIN32
#  include <windows.h>
#endif

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
using helper::vector;

OmniDriverEmu::OmniDriverEmu()
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
    , thTimer(NULL)
    , trajPts(initData(&trajPts, "trajPoints","Trajectory positions"))
    , trajTim(initData(&trajTim, "trajTiming","Trajectory timing"))
    , visu_base(NULL)
    , visu_end(NULL)
    , currentToolIndex(0)
    , isToolControlled(true)
{

    this->f_listening.setValue(true);
    //data.forceFeedback = new NullForceFeedback();
    noDevice = false;
    moveOmniBase = false;
    executeAsynchro = false;
    omniSimThreadCreated = false;
}


OmniDriverEmu::~OmniDriverEmu()
{
    if (thTimer != NULL)
        delete thTimer;
}


void OmniDriverEmu::setForceFeedbacks(vector<ForceFeedback*> ffs)
{
    data.forceFeedbacks.clear();
    for (unsigned int i=0; i<ffs.size(); i++)
        data.forceFeedbacks.push_back(ffs[i]);
    data.forceFeedbackIndice = 0;
}


void OmniDriverEmu::cleanup()
{
    sout << "OmniDriverEmu::cleanup()" << sendl;

    // If the thread is still running stop it
    if (omniSimThreadCreated)
    {
#ifndef WIN32
        int err = pthread_cancel(hapSimuThread);

        // no error: thread cancel
        if(err==0)
        {
            std::cout << "OmniDriverEmu: thread haptic cancel in cleanup" << std::endl;

        }

        // error
        else
        {
            std::cout << "OmniDriverEmu: thread not cancel in cleanup = "  << err  << std::endl;
        }
#endif
    }
}

void OmniDriverEmu::init()
{
    std::cout << "[OmniEmu] init" << std::endl;
    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState) serr << "OmniDriverEmu has no binding MechanicalState" << sendl;
    else std::cout << "[Omni] init" << std::endl;

    if(mState->getSize()<(size_t)toolCount.getValue())
        mState->resize(toolCount.getValue());
}


/**
 function that is used to emulate a haptic device by interpolating the position of the tool between several points.
*/
void *hapticSimuExecute( void *ptr )
{

    // Initialization
    OmniDriverEmu *omniDrv = (OmniDriverEmu*)ptr;
    //double timeScale = 1.0 / (double)helper::system::thread::CTime::getTicksPerSec();
    //double startTime, endTime, totalTime, realTimePrev = -1.0, realTimeAct;
    //double requiredTime = 1.0/double(omniDrv->simuFreq.getValue()) * 1.0/timeScale; // [us]
    //double timeCorrection = 0.1 * requiredTime;
    //int timeToSleep;

    // Init the "trajectory" data
    OmniDriverEmu::VecCoord pts = omniDrv->trajPts.getValue(); //sets of points use for interpolation
    helper::vector<double> tmg = omniDrv->trajTim.getValue(); //sets of "key time" for interpolation

    if (pts.empty())
    {
        std::cerr << "Bad trajectory specification : there are no points for interpolation. " << std::endl;
        return 0;
    }


    // Add a first point ( 0 0 0 0 0 0 1 ) if the first "key time" is not 0
    if (tmg[0] != 0)
    {
        pts.insert(pts.begin(), OmniDriverEmu::Coord());
        tmg.insert(tmg.begin(), 0);
    }

    unsigned int numPts = pts.size();
    unsigned int numSegs = tmg.size();

    // test if "trajectory" data are correct
    if (numSegs != numPts)
    {
        std::cerr << "Bad trajectory specification : the number of trajectory timing does not match the number of trajectory points. " << std::endl;
        return 0;
    }

    helper::vector< unsigned int > stepNum;
    // Init the Step list
    for (unsigned int i = 0; i < numPts; i++)
    {
        stepNum.push_back(tmg[i] * omniDrv->simuFreq.getValue());
    }

    /* //Igor version

    std::cout << "numSegs " << numSegs << "  --- " << numPts << std::endl;

    std::cout << pts[0].getCenter() << "///  " << pts[0].getOrientation() << std::endl;


    if (numSegs != (2*numPts - 1))  {
        std::cerr << "Bad trajectory specification " << std::endl;
        return(0);
    }
    OmniDriverEmu::VecCoord stepDiff;
    helper::vector<int> stepNum;

    unsigned int seg = 0;
    for (unsigned int np = 0; np < numPts; np++) {
        //for the point
        unsigned int n = tmg[seg]*omniDrv->simuFreq.getValue();
        stepNum.push_back(n);
        cout << "N pts = " << n << std::endl;
        OmniDriverEmu::Coord crd;
        cout << " adding  " << crd << std::endl;
        stepDiff.push_back(crd);

        //for the line
        if (np < numPts-1) {
            seg++;
            n = tmg[seg]*omniDrv->simuFreq.getValue();
            cout << "N lin = " << n << std::endl;
            stepNum.push_back(n);
            Vec3d dx = (pts[np+1].getCenter() - pts[np].getCenter())/double(n);
            helper::Quater<double> dor;  ///TODO difference for rotations!!!
            OmniDriverEmu::Coord crd(dx, dor);
            cout << "adding " << crd << std::endl;
            stepDiff.push_back(crd);
        }
        seg++;
    }

    std::cout << " stepNum = " << stepNum << std::endl;
    std::cout << " stepDiff = " << stepDiff << std::endl;
    */


    // Init data for interpolation.
    SolidTypes<double>::SpatialVector temp1, temp2;
    long long unsigned asynchroStep=0;
    double averageFreq = 0.0, minimalFreq=1e10;
    unsigned int actSeg = 0;
    unsigned int actStep = 0;
    sofa::helper::Quater<double> actualRot;
    sofa::defaulttype::Vec3d actualPos = pts[0].getCenter();


    double timeScale = 1.0 / (double)helper::system::thread::CTime::getTicksPerSec();
    double startTime, endTime, totalTime, realTimePrev = -1.0, realTimeAct;
    double requiredTime = 1.0 / (double)omniDrv->simuFreq.getValue() * 1.0 / timeScale;
    double timeCorrection = 0.1 * requiredTime;
    int timeToSleep;

    int oneTimeMessage = 0;
    // loop that updates the position tool.
    while (true)
    {
        if (omniDrv->executeAsynchro)
        {
            if (oneTimeMessage == 1)
            {
                oneTimeMessage = 0;
            }


            startTime = double(omniDrv->thTimer->getTime());


            // compute the new position and orientataion
            if (actSeg < numSegs)
            {

                if (actStep < stepNum[actSeg])
                {

                    //compute the coeff for interpolation
                    double t = ((double)(actStep-stepNum[actSeg-1]))/((double)(stepNum[actSeg]-stepNum[actSeg-1]));

                    //compute the actual position
                    actualPos = (pts[actSeg-1].getCenter())*(1-t)+(pts[actSeg].getCenter())*t;

                    //compute the actual orientation
                    actualRot.slerp(pts[actSeg-1].getOrientation(),pts[actSeg].getOrientation(),t,true);

                    actStep++;
                }
                else
                {

                    actualPos = pts[actSeg].getCenter();
                    actualRot = pts[actSeg].getOrientation();
                    actSeg++;

                }
            }
            else
            {

                std::cout << "OmniDriverEmu : End of the movement!" << std::endl;
                omniDrv->setOmniSimThreadCreated(false);
#ifndef WIN32
                pthread_exit(0);
#else
                return 0;
#endif
            }





            // Update the position of the tool
            omniDrv->data.servoDeviceData.pos = actualPos;
            omniDrv->data.servoDeviceData.quat = actualRot;
            SolidTypes<double>::Transform baseOmni_H_endOmni(actualPos * omniDrv->data.scale, actualRot);
            SolidTypes<double>::Transform world_H_virtualTool = omniDrv->data.world_H_baseOmni * baseOmni_H_endOmni * omniDrv->data.endOmni_H_virtualTool;


            // transmit the position of the tool to the force feedback
            omniDrv->data.forceFeedbackIndice= omniDrv->getCurrentToolIndex();
            // store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
            for (unsigned int i=0; i<omniDrv->data.forceFeedbacks.size(); i++)
            {
                if (omniDrv->data.forceFeedbacks[i]->indice==omniDrv->data.forceFeedbackIndice)
                {
                    omniDrv->data.forceFeedbacks[i]->computeWrench(world_H_virtualTool,temp1,temp2);
                }
            }

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

            endTime = (double)omniDrv->thTimer->getTime();  //[s]
            totalTime = (endTime - startTime);  // [us]
            timeToSleep = int( (requiredTime - totalTime) - timeCorrection); //  [us]

            if (timeToSleep > 0)
            {
#ifndef WIN32
                // Microseconds sleep
                usleep(1000000.0 * timeScale * timeToSleep);
#else
                // Milliseconds sleep
                Sleep(static_cast<DWORD>(1000.0 * timeScale * timeToSleep));
#endif
                //std::cout << "Frequency OK, computation time: " << totalTime << std::endl;
            }
            else
            {
                std::cout << "Cannot achieve desired frequency, computation too slow : " << totalTime * timeScale << " seconds for last iteration." << std::endl;
            }

        }
        else
        {
            if (oneTimeMessage == 0)
            {
                std::cout << "Running Asynchro without action" << std::endl;
                oneTimeMessage = 1;
            }
#ifndef WIN32
            usleep(10000);
#else
            Sleep(static_cast<DWORD>(10000));
#endif
        }


    }
}

void OmniDriverEmu::bwdInit()
{
    sout<<"OmniDriverEmu::bwdInit() is called"<<sendl;
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node

    // depending on toolCount, search either the first force feedback, or the feedback with indice "0"
    simulation::Node *groot = dynamic_cast<simulation::Node *>(context->getRootContext()); // access to current node

    vector<ForceFeedback*> ffs;
    groot->getTreeObjects<ForceFeedback>(&ffs);
    std::cout << "OmniDriver: "<<ffs.size()<<" ForceFeedback objects found"<<std::endl;
    setForceFeedbacks(ffs);

    setDataValue();

    copyDeviceDataCallback(&data);

    if (omniSimThreadCreated)
    {
        serr << "Emulating thread already running" << sendl;

#ifndef WIN32
        int err = pthread_cancel(hapSimuThread);

        // no error: thread cancel
        if(err==0)
        {
            std::cout << "OmniDriverEmu: thread haptic cancel" << std::endl;

        }

        // error
        else
        {
            std::cout << "thread not cancel = "  << err  << std::endl;
        }
#endif
    }
    //sout << "Not initializing phantom, starting emulating thread..." << sendl;
    //pthread_t hapSimuThread;

    if (thTimer == NULL)
        thTimer = new(helper::system::thread::CTime);

#ifndef WIN32
    if ( pthread_create( &hapSimuThread, NULL, hapticSimuExecute, (void*)this) == 0 )
    {
        std::cout << "OmniDriver : Thread created for Omni simulation" << std::endl;
        omniSimThreadCreated=true;
    }

    /* } else
        sout << "Emulating thread already running" << sendl;
        */
#else
    boost::thread hapSimuThread(hapticSimuExecute, this);
    setOmniSimThreadCreated(true);
#endif
}


void OmniDriverEmu::setDataValue()
{

    //ajout
    data.forceFeedbackIndice=0;
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

void OmniDriverEmu::reset()
{
    std::cout<<"OmniDriverEmu::reset() is called" <<std::endl;
    this->reinit();
}

void OmniDriverEmu::reinitVisual()
{
}


void OmniDriverEmu::reinit()
{
    std::cout<<"OmniDriverEmu::reinit() is called" <<std::endl;
    this->cleanup();
    this->bwdInit();
    this->reinitVisual();
    std::cout<<"OmniDriverEmu::reinit() done" <<std::endl;


    //////////////// visu_base: place the visual model of the OmniDriver


    //sofa::component::visualmodel::RigidMappedModel::VecCoord* x_rigid = visu_base->getRigidX();
    // x_rigid->resize(1);
    //(*x_rigid)[0].getOrientation() = q;
    //(*x_rigid)[0].getCenter() =  positionBase.getValue();
    //double s =
    //this->scale=Vector3(this->)

}

void OmniDriverEmu::draw(const core::visual::VisualParams *)
{
    if(omniVisu.getValue())
    {
        // compute position of the endOmni in worldframe
        defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
        defaulttype::SolidTypes<double>::Transform world_H_endOmni = data.world_H_baseOmni * baseOmni_H_endOmni ;

        visu_base = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
        visu_base->fileMesh.setValue("mesh/omni_test2.obj");
        visu_base->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_base->setColor(1.0f,1.0f,1.0f,1.0f);
        visu_base->init();
        visu_base->initVisual();
        visu_base->updateVisual();
        visu_base->applyRotation(orientationBase.getValue());
        visu_base->applyTranslation( positionBase.getValue()[0],positionBase.getValue()[1], positionBase.getValue()[2]);

        visu_end = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
        visu_end->fileMesh.setValue("mesh/stylus.obj");
        visu_end->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_end->setColor(1.0f,0.3f,0.0f,1.0f);
        visu_end->init();
        visu_end->initVisual();
        visu_end->updateVisual();
        visu_end->applyRotation(world_H_endOmni.getOrientation());
        visu_end->applyTranslation(world_H_endOmni.getOrigin()[0],world_H_endOmni.getOrigin()[1],world_H_endOmni.getOrigin()[2]);

        // draw the 2 visual models
        visu_base->drawVisual(sofa::core::visual::VisualParams::defaultInstance());
        visu_end->drawVisual(sofa::core::visual::VisualParams::defaultInstance());
    }


}

void OmniDriverEmu::copyDeviceDataCallback(OmniData *pUserData)
{
    OmniData *data = pUserData; // static_cast<OmniData*>(pUserData);
    memcpy(&data->deviceData, &data->servoDeviceData, sizeof(DeviceData));
    data->servoDeviceData.ready = true;
    data->servoDeviceData.nupdates = 0;
}


void OmniDriverEmu::stopCallback(OmniData *pUserData)
{
    OmniData *data = pUserData; // static_cast<OmniData*>(pUserData);
    data->servoDeviceData.stop = true;
}

void OmniDriverEmu::onKeyPressedEvent(core::objectmodel::KeypressedEvent * /*kpe*/)
{



}

void OmniDriverEmu::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent * /*kre*/)
{



}

void OmniDriverEmu::handleEvent(core::objectmodel::Event *event)
{


    //std::cout<<"NewEvent detected !!"<<std::endl;


    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        if (this->f_printLog.getValue())
        {
            std::cout << "Test handle event "<< std::endl;
        }

        //getData(); // copy data->servoDeviceData to gDeviceData
        //if (!simulateTranslation.getValue()) {
        copyDeviceDataCallback(&data);

        if (this->f_printLog.getValue())
        {
            std::cout << data.deviceData.ready<< std::endl;
        }

        if (data.deviceData.ready)
        {
            if (this->f_printLog.getValue())
            {
                std::cout << "Data ready, event 2"<< std::endl;
            }

            data.deviceData.quat.normalize();

            if (isToolControlled) // ignore haptic device if tool is unselected
            {

                /// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
                SolidTypes< double >::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
                SolidTypes< double >::Transform world_H_virtualTool = data.world_H_baseOmni * baseOmni_H_endOmni * data.endOmni_H_virtualTool;

                //----------------------------
                data.forceFeedbackIndice=currentToolIndex;
                // store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
                //data.forceFeedback->setReferencePosition(world_H_virtualTool);
                for (unsigned int i=0; i<data.forceFeedbacks.size(); i++)
                    if (data.forceFeedbacks[i]->indice==data.forceFeedbackIndice)
                        data.forceFeedbacks[i]->setReferencePosition(world_H_virtualTool);
                //-----------------------------


                /// TODO : SHOULD INCLUDE VELOCITY !!
                //sofa::core::objectmodel::HapticDeviceEvent omniEvent(data.deviceData.id, world_H_virtualTool.getOrigin(), world_H_virtualTool.getOrientation() , data.deviceData.m_buttonState);
                //this->getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &omniEvent);

                helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > x = *this->mState->write(core::VecCoordId::position());
                this->getContext()->getMechanicalState()->vRealloc( sofa::core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition() ); // freePosition is not allocated by default
                helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > xfree = *this->mState->write(core::VecCoordId::freePosition());


                /// FIX : check if the mechanical state is empty, if true, resize it
                /// otherwise: crash when accessing xfree[] and x[]
                if(xfree.size() == 0)
                    xfree.resize(1);
                if(x.size() == 0)
                    x.resize(1);

                if((size_t)currentToolIndex >= xfree.size() || (size_t)currentToolIndex >= x.size())
                    serr<<"currentToolIndex exceed the size of xfree/x vectors"<<std::endl;
                else
                {
                    xfree[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();
                    x[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();


                    xfree[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();
                    x[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();
                }


                sofa::simulation::Node *node = dynamic_cast<sofa::simulation::Node*> (this->getContext());
                if (node != NULL)
                {
                    sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
                    sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(node);
                }
            }
            else
            {
                data.forceFeedbackIndice = -1;
            }


            if (moveOmniBase)
            {

                std::cout<<" new positionBase = "<<positionBase_buf[0]<<std::endl;
                visu_base->applyTranslation(positionBase_buf[0]-positionBase.getValue()[0],
                        positionBase_buf[1]-positionBase.getValue()[1],
                        positionBase_buf[2]-positionBase.getValue()[2]);
                positionBase.setValue(positionBase_buf);
                setDataValue();
                //this->reinitVisual();
            }


            executeAsynchro = true;
        }
        else
            std::cout<<"data not ready"<<std::endl;
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

        // emulated haptic buttons B=btn1, N=btn2
        if (kpe->getKey()=='H' || kpe->getKey()=='h')
        {
            std::cout << "emulated button 1 pressed" << std::endl;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,
                    sofa::core::objectmodel::HapticDeviceEvent::Button1StateMask);
            simulation::Node *groot = dynamic_cast<simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }
        if (kpe->getKey()=='J' || kpe->getKey()=='j')
        {
            std::cout << "emulated button 2 pressed" << std::endl;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,
                    sofa::core::objectmodel::HapticDeviceEvent::Button2StateMask);
            simulation::Node *groot = dynamic_cast<simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }

    }
    if (dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event))
    {
        core::objectmodel::KeyreleasedEvent *kre = dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event);
        // emulated haptic buttons B=btn1, N=btn2
        if (kre->getKey()=='H' || kre->getKey()=='h'
            || kre->getKey()=='J' || kre->getKey()=='j')
        {
            std::cout << "emulated button released" << std::endl;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,0);
            simulation::Node *groot = dynamic_cast<simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }
    }
}


int OmniDriverEmuClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< OmniDriverEmu >();

SOFA_DECL_CLASS(OmniDriverEmu)


} // namespace controller

} // namespace component

} // namespace sofa
