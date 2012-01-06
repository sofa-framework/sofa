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

#include "OmniDriver.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
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

#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
//sensable namespace


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

HDCallbackCode HDCALLBACK stateCallbackOmni(void *userData)
{
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
    SolidTypes<double>::Transform baseOmni_H_endOmni(pos* data->scale, rot);
    SolidTypes<double>::Transform world_H_virtualTool = data->world_H_baseOmni * baseOmni_H_endOmni * data->endOmni_H_virtualTool;

    Vec3d world_pos_tool = world_H_virtualTool.getOrigin();
    Quat world_quat_tool = world_H_virtualTool.getOrientation();

    ///////////////// 3D rendering ////////////////
    //double fx=0.0, fy=0.0, fz=0.0;
    //if (data->forceFeedback != NULL)
    //	(data->forceFeedback)->computeForce(world_pos_tool[0], world_pos_tool[1], world_pos_tool[2], world_quat_tool[0], world_quat_tool[1], world_quat_tool[2], world_quat_tool[3], fx, fy, fz);
    //// generic computation with a 6D haptic feedback : the forceFeedback provide a force and a torque applied at point Tool but computed in the World frame
    //SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(fx,fy,fz), Vec3d(0.0,0.0,0.0));


    ///////////////// 6D rendering ////////////////
    SolidTypes<double>::SpatialVector Twist_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0)); // Todo: compute a velocity !!
    SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));

    if (data->forceFeedback != NULL)
        (data->forceFeedback)->computeWrench(world_H_virtualTool,Twist_tool_inWorld,Wrench_tool_inWorld );

    // we compute its value in the current Tool frame:
    SolidTypes<double>::SpatialVector Wrench_tool_inTool(world_quat_tool.inverseRotate(Wrench_tool_inWorld.getForce()),  world_quat_tool.inverseRotate(Wrench_tool_inWorld.getTorque())  );
    // we transport (change of application point) its value to the endOmni frame
    SolidTypes<double>::SpatialVector Wrench_endOmni_inEndOmni = data->endOmni_H_virtualTool * Wrench_tool_inTool;
    // we compute its value in the baseOmni frame
    SolidTypes<double>::SpatialVector Wrench_endOmni_inBaseOmni( baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getForce()), baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getTorque()) );

    double currentForce[3];
    currentForce[0] = Wrench_endOmni_inBaseOmni.getForce()[0] * data->forceScale;
    currentForce[1] = Wrench_endOmni_inBaseOmni.getForce()[1] * data->forceScale;
    currentForce[2] = Wrench_endOmni_inBaseOmni.getForce()[2] * data->forceScale;

    //cout << "OMNIDATA " << world_H_virtualTool.getOrigin() << " " << Wrench_tool_inWorld.getForce() << endl; // << currentForce[0] << " " << currentForce[1] << " " << currentForce[2] << endl;
    if((data->servoDeviceData.m_buttonState & HD_DEVICE_BUTTON_1) || data->permanent_feedback)
        hdSetDoublev(HD_CURRENT_FORCE, currentForce);

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
            printError(stderr, &error, "[NewOmni] Failed to initialize the device");
            return -1;
        }
        printf("[NewOmni] Found device %s\n",hdGetString(HD_DEVICE_MODEL_TYPE));

        hdEnable(HD_FORCE_OUTPUT);
        hdEnable(HD_MAX_FORCE_CLAMPING);

        // Start the servo loop scheduler.
        hdStartScheduler();
        if (HD_DEVICE_ERROR(error = hdGetError()))
        {
            printError(stderr, &error, "[NewOmni] Failed to start the scheduler");
            return -1;
        }
    }

    data.servoDeviceData.ready = false;
    data.servoDeviceData.stop = false;
    hStateHandle = hdScheduleAsynchronous( stateCallbackOmni, (void*) &data, HD_MAX_SCHEDULER_PRIORITY);

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
    , visu_base(NULL)
    , visu_end(NULL)
{

    this->f_listening.setValue(true);
    data.forceFeedback = new NullForceFeedback();
    noDevice = false;
    moveOmniBase = false;
}

OmniDriver::~OmniDriver()
{
    /*  if (visu_base)
      {
        delete visu_base;
      }
      if (visu_end)
      {
        delete visu_end;
      }
    */
}

void OmniDriver::cleanup()
{
    sout << "OmniDriver::cleanup()" << sendl;
    hdScheduleSynchronous(stopCallbackOmni, (void*) &data, HD_MIN_SCHEDULER_PRIORITY);
    //exitHandler();
    isInitialized = false;
//    delete forceFeedback;
}

void OmniDriver::setForceFeedback(ForceFeedback* ff)
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

void OmniDriver::init()
{
    using core::behavior::MechanicalState;
    mState = dynamic_cast<MechanicalState<Rigid3dTypes> *> (this->getContext()->getMechanicalState());
    if (!mState)
        serr << "OmniDriver has no binding MechanicalState" << sendl;
    else
        std::cout << "[NewOmni] init" << endl;
}


void OmniDriver::bwdInit()
{
    std::cout<<"OmniDriver::bwdInit() is called"<<std::endl;
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
    ForceFeedback *ff = context->getTreeObject<ForceFeedback>();

    if(ff)
    {
        this->setForceFeedback(ff);
    }
    //std::cerr << "setForceFeedback(ff) ok" << std::endl;
    setDataValue();
    //std::cerr << "OmniDriver::bwdInit() setDataValueOK" << std::endl;

    if(initDevice(data)==-1)
    {
        noDevice=true;
        std::cout<<"WARNING NO DEVICE"<<std::endl;
    }
}


void OmniDriver::setDataValue()
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

void OmniDriver::reset()
{
    std::cout<<"OmniDriver::reset() is called" <<std::endl;
    this->reinit();
}

void OmniDriver::reinitVisual()
{
    cout << "OmniDriver::reinitVisual() is called " << endl;
    if(visu_base!=NULL)
    {
        cout << "visu_base = " << visu_base << endl;
//        delete(visu_base);
        visu_base = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
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
        //serr<<"create visual model for OmniDriver end"<<sendl;
        cout << "visu_end = " << visu_end << endl;
//        delete(visu_end);
        visu_end = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
        visu_end->fileMesh.setValue("mesh/stylus.obj");
        visu_end->m_scale.setValue(defaulttype::Vector3(scale.getValue(),scale.getValue(),scale.getValue()));
        visu_end->setColor(1.0f,0.3f,0.0f,1.0f);
        visu_end->init();
        visu_end->initVisual();
        visu_end->updateVisual();
    }


}

void OmniDriver::reinit()
{
    std::cout<<"OmniDriver::reinit() is called" <<std::endl;
    this->cleanup();
    this->bwdInit();
    this->reinitVisual();
    std::cout<<"OmniDriver::reinit() done" <<std::endl;


//////////////// visu_base: place the visual model of the OmniDriver


    //sofa::component::visualmodel::RigidMappedModel::VecCoord* x_rigid = visu_base->getRigidX();
    // x_rigid->resize(1);
    //(*x_rigid)[0].getOrientation() = q;
    //(*x_rigid)[0].getCenter() =  positionBase.getValue();
    //double s =
    //this->scale=Vector3(this->)

}

void OmniDriver::draw()
{
    //cout << "OmniDriver::draw is called" << endl;
    if(omniVisu.getValue())
    {
        if (visu_base == NULL)
        {
            cout << "Creating visu_base" << endl;
            // create visual object
            //serr<<"create visual model for OmniDriver base"<<sendl;
            visu_base = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
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
            //serr<<"create visual model for OmniDriver end"<<sendl;
            visu_end = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
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
        visu_base->drawVisual(sofa::core::visual::VisualParams::defaultInstance());
        visu_end->drawVisual(sofa::core::visual::VisualParams::defaultInstance());
    }
}

void OmniDriver::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kpe)
{



}

void OmniDriver::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kre)
{

    //omniVisu.setValue(false);

}

void OmniDriver::handleEvent(core::objectmodel::Event *event)
{

    //std::cout<<"NewEvent detected !!"<<std::endl;


    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        //getData(); // copy data->servoDeviceData to gDeviceData
        hdScheduleSynchronous(copyDeviceDataCallbackOmni, (void *) &data, HD_MIN_SCHEDULER_PRIORITY);
        if (data.deviceData.ready)
        {
            data.deviceData.quat.normalize();
            //sout << "driver is working ! " << data->servoDeviceData.transform[12+0] << endl;


            /// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
            SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);
            SolidTypes<double>::Transform world_H_virtualTool = data.world_H_baseOmni * baseOmni_H_endOmni * data.endOmni_H_virtualTool;


            // store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
            data.forceFeedback->setReferencePosition(world_H_virtualTool);

            /// TODO : SHOULD INCLUDE VELOCITY !!

            helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > x = *this->mState->write(core::VecCoordId::position());
            helper::WriteAccessor<Data<helper::vector<RigidCoord<3,double> > > > xfree = *this->mState->write(core::VecCoordId::freePosition());

            xfree[0].getCenter() = world_H_virtualTool.getOrigin();
            x[0].getCenter() = world_H_virtualTool.getOrigin();

            xfree[0].getOrientation() = world_H_virtualTool.getOrientation();
            x[0].getOrientation() = world_H_virtualTool.getOrientation();

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



    }



}

int OmniDriverClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< OmniDriver >();

SOFA_DECL_CLASS(OmniDriver)


} // namespace controller

} // namespace component

} // namespace sofa
