/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/type/Quat.h>

#include <sofa/core/visual/VisualParams.h>

////force feedback
#include <sofa/component/haptics/ForceFeedback.h>
#include <sofa/component/haptics/NullForceFeedback.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/simulation/PauseEvent.h>
#include <sofa/simulation/Node.h>

#include <sofa/simulation/Node.h>
#include <cstring>

#include <sofa/gl/component/rendering3d/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

#include <chrono>

#include <sofa/simulation/UpdateMappingVisitor.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor;

#include <sofa/helper/rmath.h>

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::type;
using namespace sofa::defaulttype;
using namespace core::behavior;
using type::vector;

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
    , thTimer(nullptr)
    , trajPts(initData(&trajPts, "trajPoints","Trajectory positions"))
    , trajTim(initData(&trajTim, "trajTiming","Trajectory timing"))
    , visu_base(nullptr)
    , visu_end(nullptr)
    , currentToolIndex(0)
    , isToolControlled(true)
{
    this->f_listening.setValue(true);
    noDevice = false;
    moveOmniBase = false;
    executeAsynchro = false;
    omniSimThreadCreated = false;
    m_terminate = true;
}


OmniDriverEmu::~OmniDriverEmu()
{
    if (thTimer != nullptr)
        delete thTimer;
}


void OmniDriverEmu::setForceFeedbacks(vector<haptics::ForceFeedback*> ffs)
{
    data.forceFeedbacks.clear();
    for (unsigned int i=0; i<ffs.size(); i++)
        data.forceFeedbacks.push_back(ffs[i]);
    data.forceFeedbackIndice = 0;
}


void OmniDriverEmu::cleanup()
{
    if (m_terminate == false && omniSimThreadCreated)
    {
        m_terminate = true;
        hapSimuThread.join();
        omniSimThreadCreated = false;
        msg_info() << "Haptic thread has been cancelled in cleanup without error.";
    }
}

void OmniDriverEmu::init()
{
    msg_info() << "[OmniEmu] init" ;
    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        msg_warning() << "OmniDriverEmu has no binding MechanicalState.";
    else
        msg_info() << "[Omni] init" ;

    if(mState->getSize()<toolCount.getValue())
        mState->resize(toolCount.getValue());
}


/**
 function that is used to emulate a haptic device by interpolating the position of the tool between several points.
*/
void hapticSimuExecute(std::atomic<bool>& terminate, void *ptr )
{
    assert(ptr!=nullptr);

    // Initialization
    OmniDriverEmu *omniDrv = static_cast<OmniDriverEmu*>(ptr);

    // Init the "trajectory" data
    OmniDriverEmu::VecCoord pts = omniDrv->trajPts.getValue(); //sets of points use for interpolation
    type::vector<double> tmg = omniDrv->trajTim.getValue(); //sets of "key time" for interpolation

    if (pts.empty())
    {
        msg_error(omniDrv) << "Bad trajectory specification : there are no points for interpolation. ";
        return;
    }

    // Add a first point ( 0 0 0 0 0 0 1 ) if the first "key time" is not 0
    if (sofa::helper::isEqual(tmg[0], 0.0))
    {
        pts.insert(pts.begin(), OmniDriverEmu::Coord());
        tmg.insert(tmg.begin(), 0);
    }

    size_t numPts = pts.size();
    size_t numSegs = tmg.size();

    // test if "trajectory" data are correct
    if (numSegs != numPts)
    {
        msg_error(omniDrv) << "Bad trajectory specification : the number of trajectory timing does not match the number of trajectory points. ";
        return;
    }

    type::vector< unsigned int > stepNum;
    // Init the Step list
    for (unsigned int i = 0; i < numPts; i++)
    {
        stepNum.push_back(tmg[i] * omniDrv->simuFreq.getValue());
    }

    // Init data for interpolation.
    SolidTypes<double>::SpatialVector temp1, temp2;
    long long unsigned asynchroStep=0;
    double averageFreq = 0.0, minimalFreq=1e10;
    unsigned int actSeg = 0;
    unsigned int actStep = 0;
    sofa::type::Quat<double> actualRot;
    sofa::type::Vec3d actualPos = pts[0].getCenter();

    double timeScale = 1.0 / (double)helper::system::thread::CTime::getTicksPerSec();
    double startTime, endTime, totalTime, realTimePrev = -1.0, realTimeAct;
    double requiredTime = 1.0 / (double)omniDrv->simuFreq.getValue() * 1.0 / timeScale;
    double timeCorrection = 0.1 * requiredTime;
    int timeToSleep;

    int oneTimeMessage = 0;
    // loop that updates the position tool.
    while (!terminate)
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

                if (actSeg > 0 && actStep < stepNum[actSeg])
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
                msg_info(omniDrv) << "End of the movement!" ;
                return;
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
                if (omniDrv->data.forceFeedbacks[i]->d_indice.getValue()==omniDrv->data.forceFeedbackIndice)
                {
                    omniDrv->data.forceFeedbacks[i]->computeWrench(world_H_virtualTool,temp1,temp2);
                }
            }

            realTimeAct = double(omniDrv->thTimer->getTime());

            if (asynchroStep > 0)
            {
                double realFreq = 1.0/( (realTimeAct - realTimePrev)*timeScale );
                averageFreq += realFreq;
                if (realFreq < minimalFreq)
                    minimalFreq = realFreq;

                if ( ((asynchroStep+1) % 1000) == 0)
                {
                    msg_info(omniDrv) << "Average frequency of the loop = " << averageFreq/double(asynchroStep) << " Hz "
                                      << "Minimal frequency of the loop = " << minimalFreq << " Hz " ;
                }
            }

            realTimePrev = realTimeAct;
            asynchroStep++;

            endTime = (double)omniDrv->thTimer->getTime();  //[s]
            totalTime = (endTime - startTime);  // [us]
            timeToSleep = int( ((requiredTime - totalTime) - timeCorrection) ); //  [us]

            if (timeToSleep > 0)
            {
                std::this_thread::sleep_for(std::chrono::seconds(int(timeToSleep * timeScale)));
            }
            else
            {
                msg_info(omniDrv) << "Cannot achieve desired frequency, computation too slow : " << totalTime * timeScale << " seconds for last iteration.";
            }
        }
        else
        {
            if (oneTimeMessage == 0)
            {
                msg_info(omniDrv) << "Running Asynchro without action" ;
                oneTimeMessage = 1;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void OmniDriverEmu::bwdInit()
{
    msg_info()<<"OmniDriverEmu::bwdInit() is called";
    sofa::simulation::Node *context = dynamic_cast<sofa::simulation::Node *>(this->getContext()); // access to current node

    // depending on toolCount, search either the first force feedback, or the feedback with indice "0"
    sofa::simulation::Node *groot = dynamic_cast<sofa::simulation::Node *>(context->getRootContext()); // access to current node

    vector<haptics::ForceFeedback*> ffs;
    groot->getTreeObjects<haptics::ForceFeedback>(&ffs);
    msg_info() << "OmniDriver: "<<ffs.size()<<" ForceFeedback objects found";
    setForceFeedbacks(ffs);

    setDataValue();

    copyDeviceDataCallback(&data);

    if (omniSimThreadCreated)
    {
        msg_warning() << "Emulating thread already running" ;
        m_terminate = false;
        cleanup();
    }

    if (thTimer == nullptr)
        thTimer = new(helper::system::thread::CTime);

    m_terminate = false;
    hapSimuThread = std::thread(hapticSimuExecute, std::ref(this->m_terminate), this);
    setOmniSimThreadCreated(true);
    msg_info() << "OmniDriver : Thread created for Omni simulation.";
}


void OmniDriverEmu::setDataValue()
{
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

void OmniDriverEmu::reinit()
{
    msg_info()<<"OmniDriverEmu::reinit() is called";
    this->cleanup();
    this->bwdInit();
    msg_info()<<"OmniDriverEmu::reinit() done";
}

void OmniDriverEmu::draw(const core::visual::VisualParams *)
{
    using sofa::gl::component::rendering3d::OglModel;
    if(omniVisu.getValue())
    {
        static bool isInited=false;
        if(!isInited)
        {
        // compute position of the endOmni in worldframe
        defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
        defaulttype::SolidTypes<double>::Transform world_H_endOmni = data.world_H_baseOmni * baseOmni_H_endOmni ;

        visu_base = sofa::core::objectmodel::New<OglModel>();
        visu_base->fileMesh.setValue("mesh/omni_test2.obj");
        visu_base->m_scale.setValue(type::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_base->setColor(1.0f,1.0f,1.0f,1.0f);
        visu_base->init();
        visu_base->initVisual();
        visu_base->updateVisual();
        visu_base->applyRotation(orientationBase.getValue());
        visu_base->applyTranslation( positionBase.getValue()[0],positionBase.getValue()[1], positionBase.getValue()[2]);

        visu_end = sofa::core::objectmodel::New<OglModel>();
        visu_end->fileMesh.setValue("mesh/stylus.obj");
        visu_end->m_scale.setValue(type::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_end->setColor(1.0f,0.3f,0.0f,1.0f);
        visu_end->init();
        visu_end->initVisual();
        visu_end->updateVisual();
        visu_end->applyRotation(world_H_endOmni.getOrientation());
        visu_end->applyTranslation(world_H_endOmni.getOrigin()[0],world_H_endOmni.getOrigin()[1],world_H_endOmni.getOrigin()[2]);
        isInited=true;
        }

        // draw the 2 visual models
        visu_base->doDrawVisual(sofa::core::visual::VisualParams::defaultInstance());
        visu_end->doDrawVisual(sofa::core::visual::VisualParams::defaultInstance());
    }
}

void OmniDriverEmu::copyDeviceDataCallback(OmniData *pUserData)
{
    OmniData *data = pUserData;
    memcpy(&data->deviceData, &data->servoDeviceData, sizeof(DeviceData));
    data->servoDeviceData.ready = true;
    data->servoDeviceData.nupdates = 0;
}

void OmniDriverEmu::stopCallback(OmniData *pUserData)
{
    OmniData *data = pUserData;
    data->servoDeviceData.stop = true;
}

void OmniDriverEmu::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        msg_info() << "Test handle event ";

        copyDeviceDataCallback(&data);

        msg_info() << data.deviceData.ready;

        if (data.deviceData.ready)
        {
            msg_info() << "Data ready, event 2";

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
                    if (data.forceFeedbacks[i]->d_indice.getValue()==data.forceFeedbackIndice)
                        data.forceFeedbacks[i]->setReferencePosition(world_H_virtualTool);

                //-----------------------------
                //TODO : SHOULD INCLUDE VELOCITY !!
                //sofa::core::objectmodel::HapticDeviceEvent omniEvent(data.deviceData.id, world_H_virtualTool.getOrigin(), world_H_virtualTool.getOrientation() , data.deviceData.m_buttonState);
                //this->getContext()->propagateEvent(sofa::core::ExecParams::defaultInstance(), &omniEvent);
                helper::WriteAccessor<Data<type::vector<RigidCoord<3,double> > > > x = *this->mState->write(core::VecCoordId::position());
                this->getContext()->getMechanicalState()->vRealloc( sofa::core::MechanicalParams::defaultInstance(), core::VecCoordId::freePosition() ); // freePosition is not allocated by default
                helper::WriteAccessor<Data<type::vector<RigidCoord<3,double> > > > xfree = *this->mState->write(core::VecCoordId::freePosition());

                /// FIX : check if the mechanical state is empty, if true, resize it
                /// otherwise: crash when accessing xfree[] and x[]
                if(xfree.size() == 0)
                    xfree.resize(1);
                if(x.size() == 0)
                    x.resize(1);

                if((size_t)currentToolIndex >= xfree.size() || (size_t)currentToolIndex >= x.size())
                    msg_warning()<<"currentToolIndex exceed the size of xfree/x vectors";
                else
                {
                    xfree[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();
                    x[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();

                    xfree[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();
                    x[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();
                }

                sofa::simulation::Node *node = dynamic_cast<sofa::simulation::Node*> (this->getContext());
                if (node != nullptr)
                {
                    MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
                    sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(node);
                }
            }
            else
            {
                data.forceFeedbackIndice = -1;
            }

            if (moveOmniBase)
            {
                msg_info()<<" new positionBase = "<<positionBase_buf[0];
                visu_base->applyTranslation(positionBase_buf[0]-positionBase.getValue()[0],
                        positionBase_buf[1]-positionBase.getValue()[1],
                        positionBase_buf[2]-positionBase.getValue()[2]);
                positionBase.setValue(positionBase_buf);
                setDataValue();
            }

            executeAsynchro = true;
        }
        else
            msg_info()<<"data not ready";
    }

    if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    {
        core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
        if (kpe->getKey()=='Z' ||kpe->getKey()=='z' )
        {
            moveOmniBase = !moveOmniBase;
            msg_info()<<"key z detected ";
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
            msg_info() << "emulated button 1 pressed";
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,
                                                             sofa::core::objectmodel::HapticDeviceEvent::Button1StateMask);
            sofa::simulation::Node *groot = dynamic_cast<sofa::simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }
        if (kpe->getKey()=='J' || kpe->getKey()=='j')
        {
            std::cout << "emulated button 2 pressed" << std::endl;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,
                                                             sofa::core::objectmodel::HapticDeviceEvent::Button2StateMask);
            sofa::simulation::Node *groot = dynamic_cast<sofa::simulation::Node *>(getContext()->getRootContext()); // access to current node
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
            msg_info() << "emulated button released" ;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,0);
            sofa::simulation::Node *groot = dynamic_cast<sofa::simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }
    }
}


static int OmniDriverEmuClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< OmniDriverEmu >();

} // namespace controller

} // namespace component

} // namespace sofa
