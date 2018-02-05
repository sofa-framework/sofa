/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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

#include "OmniDriver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
//
////force feedback
#include <SofaHaptics/ForceFeedback.h>
#include <SofaHaptics/NullForceFeedback.h>
//
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
//
#include <sofa/simulation/Node.h>
#include <cstring>

#include <SofaOpenglVisual/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
//sensable namespace
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/helper/system/thread/CTime.h>
#ifdef SOFA_HAVE_BOOST
#include <boost/thread.hpp>
#endif

namespace sofa
{

namespace component
{

namespace controller
{

using namespace sofa::defaulttype;
using namespace core::behavior;
using namespace sofa::defaulttype;

//static DeviceData gServoDeviceData;
//static DeviceData deviceData;
//static DeviceData previousData;
static HHD hHD = HD_INVALID_HANDLE ;
static bool isInitialized = false;
static HDSchedulerHandle hStateHandle = HD_INVALID_HANDLE;

static sofa::helper::system::atomic<int> doUpdate;

void printError(FILE *stream, const HDErrorInfo *error,
        const char *message)
{
    fprintf(stream, "%s\n", hdGetErrorString(error->errorCode));
    fprintf(stream, "HHD: %X\n", error->hHD);
    fprintf(stream, "Error Code: %X\n", error->errorCode);
    fprintf(stream, "Internal Error Code: %d\n", error->internalErrorCode);
    fprintf(stream, "Message: %s\n", message);
}

bool isSchedulerError(const HDErrorInfo *error)
{
    switch (error->errorCode)
    {
    case HD_COMM_ERROR:
    case HD_COMM_CONFIG_ERROR:
    case HD_TIMER_ERROR:
    case HD_INVALID_PRIORITY:
    case HD_SCHEDULER_FULL:
        return true;

    default:
        return false;
    }
}

HDCallbackCode HDCALLBACK copyDeviceDataCallbackOmni(void *userData);

HDCallbackCode HDCALLBACK stateCallbackOmni(void *userData)
{


    if(doUpdate)
    {
        copyDeviceDataCallbackOmni(userData);
        doUpdate.dec(); // set to 0
    }


    //cout << "OmniDriver::stateCallback BEGIN" << endl;
    OmniData* data = static_cast<OmniData*>(userData);
    //FIXME : Apparenlty, this callback is run before the mechanical state initialisation. I've found no way to know whether the mechcanical state is initialized or not, so i wait ...
    //static int wait = 0;

    if (data->servoDeviceData.stop)
    {
        //cout << ""
        return HD_CALLBACK_DONE;
    }

    if (!data->servoDeviceData.ready)
    {
        return HD_CALLBACK_CONTINUE;
    }

    HHD hapticHD = hdGetCurrentDevice();
    hdBeginFrame(hapticHD);

    data->servoDeviceData.id = hapticHD;

    //static int renderForce = true;

    // Retrieve the current button(s).
    hdGetIntegerv(HD_CURRENT_BUTTONS, &data->servoDeviceData.m_buttonState);

    hdGetDoublev(HD_CURRENT_POSITION, data->servoDeviceData.m_devicePosition);
    // Get the column major transform
    HDdouble transform[16];
    hdGetDoublev(HD_CURRENT_TRANSFORM, transform);

    // get Position and Rotation from transform => put in servoDeviceData
    Mat3x3d mrot;
    Quat rot;
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            mrot[i][j] = transform[j*4+i];

    rot.fromMatrix(mrot);
    rot.normalize();

    double factor = 0.001;
    Vec3d pos(transform[12+0]*factor, transform[12+1]*factor, transform[12+2]*factor); // omni pos is in mm => sofa simulation are in meters by default
    data->servoDeviceData.pos=pos;

    // verify that the quaternion does not flip:
    if ((rot[0]*data->servoDeviceData.quat[0]+rot[1]*data->servoDeviceData.quat[1]+rot[2]*data->servoDeviceData.quat[2]+rot[3]*data->servoDeviceData.quat[3]) < 0)
        for (int i=0; i<4; i++)
            rot[i] *= -1;

    data->servoDeviceData.quat[0] = rot[0];
    data->servoDeviceData.quat[1] = rot[1];
    data->servoDeviceData.quat[2] = rot[2];
    data->servoDeviceData.quat[3] = rot[3];


    /// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
    sofa::defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(pos* data->scale, rot);
    sofa::defaulttype::SolidTypes<double>::Transform world_H_virtualTool = data->world_H_baseOmni * baseOmni_H_endOmni * data->endOmni_H_virtualTool;

    Vec3d world_pos_tool = world_H_virtualTool.getOrigin();
    Quat world_quat_tool = world_H_virtualTool.getOrientation();

    ///////////////// 3D rendering ////////////////
    //double fx=0.0, fy=0.0, fz=0.0;
    //if (data->forceFeedback != NULL)
    //	(data->forceFeedback)->computeForce(world_pos_tool[0], world_pos_tool[1], world_pos_tool[2], world_quat_tool[0], world_quat_tool[1], world_quat_tool[2], world_quat_tool[3], fx, fy, fz);
    //// generic computation with a 6D haptic feedback : the forceFeedback provide a force and a torque applied at point Tool but computed in the World frame
    //SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(fx,fy,fz), Vec3d(0.0,0.0,0.0));


    ///////////////// 6D rendering ////////////////
    sofa::defaulttype::SolidTypes<double>::SpatialVector Twist_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0)); // Todo: compute a velocity !!
    sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));

    // which forcefeedback ?
    ForceFeedback* ff = 0;
    for (int i=0; i<data->forceFeedbacks.size() && !ff; i++)
        if (data->forceFeedbacks[i]->indice==data->forceFeedbackIndice)
            ff = data->forceFeedbacks[i];

    if (ff != NULL)
        ff->computeWrench(world_H_virtualTool,Twist_tool_inWorld,Wrench_tool_inWorld );

    // we compute its value in the current Tool frame:
    sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_tool_inTool(world_quat_tool.inverseRotate(Wrench_tool_inWorld.getForce()),  world_quat_tool.inverseRotate(Wrench_tool_inWorld.getTorque())  );
    // we transport (change of application point) its value to the endOmni frame
    sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_endOmni_inEndOmni = data->endOmni_H_virtualTool * Wrench_tool_inTool;
    // we compute its value in the baseOmni frame
    sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_endOmni_inBaseOmni( baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getForce()), baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getTorque()) );

    double currentForce[3];
    currentForce[0] = Wrench_endOmni_inBaseOmni.getForce()[0] * data->forceScale;
    currentForce[1] = Wrench_endOmni_inBaseOmni.getForce()[1] * data->forceScale;
    currentForce[2] = Wrench_endOmni_inBaseOmni.getForce()[2] * data->forceScale;

//    if (Wrench_endOmni_inBaseOmni.getForce().norm() > 0.1)
//        printf("wrench = %f\n",Wrench_endOmni_inBaseOmni.getForce().norm());

    //cout << "OMNIDATA " << world_H_virtualTool.getOrigin() << " " << Wrench_tool_inWorld.getForce() << endl; // << currentForce[0] << " " << currentForce[1] << " " << currentForce[2] << endl;
    if((data->servoDeviceData.m_buttonState & HD_DEVICE_BUTTON_1) || data->permanent_feedback)
        hdSetDoublev(HD_CURRENT_FORCE, currentForce);
    else
    {
        // reset force feedback
        currentForce[0] = 0.0;
        currentForce[1] = 0.0;
        currentForce[2] = 0.0;
        hdSetDoublev(HD_CURRENT_FORCE, currentForce);
    }




    ++data->servoDeviceData.nupdates;
    hdEndFrame(hapticHD);

    /* HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
    	printError(stderr, &error, "Error during scheduler callback");
    	if (isSchedulerError(&error))
    	{
    		return HD_CALLBACK_DONE;
    	}
           }*/
    /*
     	OmniX = data->servoDeviceData.transform[12+0]*0.1;
    	OmniY =	data->servoDeviceData.transform[12+1]*0.1;
    	OmniZ =	data->servoDeviceData.transform[12+2]*0.1;
    */

    //cout << "OmniDriver::stateCallback END" << endl;
    return HD_CALLBACK_CONTINUE;
}

void exitHandlerOmni()
{
    hdStopScheduler();
    hdUnschedule(hStateHandle);
    /*
        if (hHD != HD_INVALID_HANDLE)
        {
            hdDisableDevice(hHD);
            hHD = HD_INVALID_HANDLE;
        }
    */
}


HDCallbackCode HDCALLBACK copyDeviceDataCallbackOmni(void *pUserData)
{
    OmniData *data = static_cast<OmniData*>(pUserData);
    memcpy(&data->deviceData, &data->servoDeviceData, sizeof(DeviceData));
    data->servoDeviceData.nupdates = 0;
    data->servoDeviceData.ready = true;


    return HD_CALLBACK_DONE;
}

HDCallbackCode HDCALLBACK stopCallbackOmni(void *pUserData)
{
    OmniData *data = static_cast<OmniData*>(pUserData);
    data->servoDeviceData.stop = true;
    return HD_CALLBACK_DONE;
}

/**
 * Sets up the device,
 */
int OmniDriver::initDevice(OmniData& data)
{
    if (isInitialized) return 0;
    isInitialized = true;

    data.deviceData.quat[0] = 0;
    data.deviceData.quat[1] = 0;
    data.deviceData.quat[2] = 0;
    data.deviceData.quat[3] = 1;

    data.servoDeviceData.quat[0] = 0;
    data.servoDeviceData.quat[1] = 0;
    data.servoDeviceData.quat[2] = 0;
    data.servoDeviceData.quat[3] = 1;

    HDErrorInfo error;
    // Initialize the device, must be done before attempting to call any hd functions.
    if (hHD == HD_INVALID_HANDLE)
    {
        hHD = hdInitDevice(HD_DEFAULT_DEVICE);
        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            printError(stderr, &error, "[Omni] Failed to initialize the device");
            return -1;
        }
        printf("[Omni] Found device %s\n",hdGetString(HD_DEVICE_MODEL_TYPE));

        hdEnable(HD_FORCE_OUTPUT);
        hdEnable(HD_MAX_FORCE_CLAMPING);

        // Start the servo loop scheduler.
        doUpdate = 0;
        hdStartScheduler();
        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            printError(stderr, &error, "[Omni] Failed to start the scheduler");
            return -1;
        }
    }

    data.servoDeviceData.ready = false;
    data.servoDeviceData.stop = false;
    hStateHandle = hdScheduleAsynchronous( stateCallbackOmni, (void*) &data, HD_MIN_SCHEDULER_PRIORITY);

    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        printError(stderr, &error, "Failed to initialize haptic device");
        fprintf(stderr, "\nPress any key to quit.\n");
        getchar();
        exit(-1);
    }

    return 0;
}

OmniDriver::OmniDriver()
    : scale(initData(&scale, 1.0, "scale","Default scale applied to the Phantom Coordinates. "))
    , forceScale(initData(&forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
    , positionBase(initData(&positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
    , orientationBase(initData(&orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
    , positionTool(initData(&positionTool, Vec3d(0,0,0), "positionTool","Position of the tool in the omni end effector frame"))
    , orientationTool(initData(&orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool in the omni end effector frame"))
    , permanent(initData(&permanent, false, "permanent" , "Apply the force feedback permanently"))
    , omniVisu(initData(&omniVisu, false, "omniVisu", "Visualize the position of the interface in the virtual scene"))
    , toolSelector(initData(&toolSelector, false, "toolSelector", "Switch tools with 2nd button"))
    , toolCount(initData(&toolCount, 1, "toolCount", "Number of tools to switch between"))
    , visu_base(NULL)
    , visu_end(NULL)
    , currentToolIndex(0)
    , isToolControlled(true)
{
    this->f_listening.setValue(true);
    //data.forceFeedback = new NullForceFeedback();
    noDevice = false;
    moveOmniBase = false;
}

OmniDriver::~OmniDriver() {}

void OmniDriver::cleanup()
{
    sout << "OmniDriver::cleanup()" << sendl;
    hdScheduleSynchronous(stopCallbackOmni, (void*) &data, HD_MAX_SCHEDULER_PRIORITY);
    isInitialized = false;
}

void OmniDriver::setForceFeedbacks(vector<ForceFeedback*> ffs)
{
    data.forceFeedbacks.clear();
    for (int i=0; i<ffs.size(); i++)
        data.forceFeedbacks.push_back(ffs[i]);
    data.forceFeedbackIndice = 0;
}

void OmniDriver::init()
{
    using core::behavior::MechanicalState;
    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState) serr << "OmniDriver has no binding MechanicalState" << sendl;
    else sout << "[Omni] init" << sendl;

    if(mState->getSize()<toolCount.getValue())
        mState->resize(toolCount.getValue());
}

void OmniDriver::bwdInit()
{
    sout<<"OmniDriver::bwdInit()"<<sendl;
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node

    // depending on toolCount, search either the first force feedback, or the feedback with indice "0"
    simulation::Node *groot = dynamic_cast<simulation::Node *>(context->getRootContext()); // access to current node

    vector<ForceFeedback*> ffs;
    groot->getTreeObjects<ForceFeedback>(&ffs);
    sout << ffs.size()<<" ForceFeedback objects found"<<sendl;
    setForceFeedbacks(ffs);
    setDataValue();
    if(initDevice(data)==-1)
    {
        noDevice=true;
        serr<<"NO DEVICE"<<sendl;
    }
}

void OmniDriver::setDataValue()
{
    data.forceFeedbackIndice=0;
    data.scale = scale.getValue();
    data.forceScale = forceScale.getValue();
    Quat q = orientationBase.getValue();
    q.normalize();
    orientationBase.setValue(q);
    data.world_H_baseOmni.set( positionBase.getValue(), q);
    q=orientationTool.getValue();
    q.normalize();
    data.endOmni_H_virtualTool.set(positionTool.getValue(), q);
    data.permanent_feedback = permanent.getValue();
}

void OmniDriver::reset()
{
    sout<<"OmniDriver::reset()" <<sendl;
    this->reinit();
}

void OmniDriver::reinitVisual() {}

void OmniDriver::reinit()
{
    this->bwdInit();
    this->reinitVisual();
}

void OmniDriver::draw(const core::visual::VisualParams* vparam){
	draw();
}

void OmniDriver::draw()
{
    if(omniVisu.getValue())
    {
        // compute position of the endOmni in worldframe
        sofa::defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
        sofa::defaulttype::SolidTypes<double>::Transform world_H_endOmni = data.world_H_baseOmni * baseOmni_H_endOmni ;

        visu_base = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
        visu_base->fileMesh.setValue("mesh/omni_test2.obj");
		visu_base->name.setValue("BaseOmni");
        visu_base->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_base->setColor(1.0f,1.0f,1.0f,1.0f);
        visu_base->init();
        visu_base->initVisual();
        visu_base->updateVisual();
        visu_base->applyRotation(orientationBase.getValue());
        visu_base->applyTranslation( positionBase.getValue()[0],positionBase.getValue()[1], positionBase.getValue()[2]);

        visu_end = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
        visu_end->fileMesh.setValue("mesh/stylus.obj");
		visu_end->name.setValue("Stylus");
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

void OmniDriver::onKeyPressedEvent(core::objectmodel::KeypressedEvent * /*kpe*/) {}

void OmniDriver::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent * /*kre*/) {}

void OmniDriver::handleEvent(core::objectmodel::Event *event)
{

    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        sofa::helper::AdvancedTimer::stepBegin("OmniDriver::1");
        //hdScheduleSynchronous(copyDeviceDataCallbackOmni, (void *) &data, HD_MAX_SCHEDULER_PRIORITY);

        doUpdate.inc(); // set to 1
        while(doUpdate)
        {
#ifdef SOFA_HAVE_BOOST
            boost::thread::yield();
#else
            sofa::helper::system::thread::CTime::sleep(0);
#endif
        }

        sofa::helper::AdvancedTimer::stepEnd("OmniDriver::1");
        if (data.deviceData.ready)
        {
            sofa::helper::AdvancedTimer::stepBegin("OmniDriver::2");
            data.deviceData.quat.normalize();

            /// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES

            if (isToolControlled) // ignore haptic device if tool is unselected
            {
                sofa::defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
                sofa::defaulttype::SolidTypes<double>::Transform world_H_virtualTool = data.world_H_baseOmni * baseOmni_H_endOmni * data.endOmni_H_virtualTool;
                sofa::helper::AdvancedTimer::stepEnd("OmniDriver::2");

                sofa::helper::AdvancedTimer::stepBegin("OmniDriver::3");
                // store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
                data.forceFeedbackIndice=currentToolIndex;
                // which forcefeedback ?
                for (int i=0; i<data.forceFeedbacks.size(); i++)
                    if (data.forceFeedbacks[i]->indice==data.forceFeedbackIndice)
                        data.forceFeedbacks[i]->setReferencePosition(world_H_virtualTool);

                /// TODO : SHOULD INCLUDE VELOCITY !!

                helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > x = *this->mState->write(core::VecCoordId::position());
                helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > xfree = *this->mState->write(core::VecCoordId::freePosition());
                sofa::helper::AdvancedTimer::stepEnd("OmniDriver::3");

                sofa::helper::AdvancedTimer::stepBegin("OmniDriver::4");
                xfree[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();
                x[currentToolIndex].getCenter() = world_H_virtualTool.getOrigin();

                //      std::cout << world_H_virtualTool << std::endl;

                xfree[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();
                x[currentToolIndex].getOrientation() = world_H_virtualTool.getOrientation();

                sofa::simulation::Node *node = dynamic_cast<sofa::simulation::Node*> (this->getContext());
                if (node)
                {
                    sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
                    sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(node);
                }

            }
            else
            {
                data.forceFeedbackIndice = -1;
            }



            sofa::helper::AdvancedTimer::stepEnd("OmniDriver::4");
            sofa::helper::AdvancedTimer::stepBegin("OmniDriver::5");

            // launch events on buttons changes
            static bool btn1 = false;
            static bool btn2 = false;
            bool newBtn1 = 0!=(data.deviceData.m_buttonState & HD_DEVICE_BUTTON_1);
            bool newBtn2 = 0!=(data.deviceData.m_buttonState & HD_DEVICE_BUTTON_2);
            sofa::helper::AdvancedTimer::stepEnd("OmniDriver::5");

            // special case: btn2 is mapped to tool selection if "toolSelector" is used
            if (toolSelector.getValue() && btn2!=newBtn2)
            {
                btn2 = newBtn2;
                isToolControlled = !btn2;
                if (isToolControlled)
                    currentToolIndex = (currentToolIndex+1)%toolCount.getValue();
            }

            if (btn1!=newBtn1 || (!toolSelector.getValue() && btn2!=newBtn2))
            {
                sofa::helper::AdvancedTimer::stepBegin("OmniDriver::6");
                btn1 = newBtn1;
                btn2 = newBtn2;
                unsigned char buttonState = 0;
                if(btn1) buttonState |= sofa::core::objectmodel::HapticDeviceEvent::Button1StateMask;
                if(!toolSelector.getValue() && btn2) buttonState |= sofa::core::objectmodel::HapticDeviceEvent::Button2StateMask;
                Vector3 dummyVector;
                Quat dummyQuat;
                sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,buttonState);
                simulation::Node *groot = dynamic_cast<simulation::Node *>(getContext()->getRootContext()); // access to current node
                groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
                sofa::helper::AdvancedTimer::stepEnd("OmniDriver::6");
            }


        }
        else sout<<"data not ready"<<sendl;


    }

    if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    {
        core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
        if (kpe->getKey()=='Z' ||kpe->getKey()=='z' )
        {
            moveOmniBase = !moveOmniBase;
            sout<<"key z detected "<<sendl;
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
            sout << "emulated button 1 pressed" << sendl;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,
                    sofa::core::objectmodel::HapticDeviceEvent::Button1StateMask);
            simulation::Node *groot = dynamic_cast<simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }
        if (kpe->getKey()=='J' || kpe->getKey()=='j')
        {
            sout << "emulated button 2 pressed" << sendl;
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
            sout << "emulated button released" << sendl;
            Vector3 dummyVector;
            Quat dummyQuat;
            sofa::core::objectmodel::HapticDeviceEvent event(currentToolIndex,dummyVector,dummyQuat,0);
            simulation::Node *groot = dynamic_cast<simulation::Node *>(getContext()->getRootContext()); // access to current node
            groot->propagateEvent(core::ExecParams::defaultInstance(), &event);
        }
    }
}

int OmniDriverClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< OmniDriver >();

SOFA_DECL_CLASS(OmniDriver)


} // namespace controller

} // namespace component

} // namespace sofa
