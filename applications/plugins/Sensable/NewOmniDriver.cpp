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

#include "NewOmniDriver.h"

#include <sofa/core/ObjectFactory.h>
//#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>

#include <sofa/helper/system/thread/CTime.h>
#ifdef SOFA_HAVE_BOOST
#include <boost/thread.hpp>
#endif

//sensable namespace

using std::cout;
using std::endl;

double prevTime;
bool frame;
bool visuCreation;

namespace sofa
{

namespace component
{

namespace controller
{

const char* NewOmniDriver::visualNodeNames[NVISUALNODE] =
{
    "stylus",
    "joint 2",
    "joint 1",
    "arm 2",
    "arm 1",
    "joint 0",
    "base",
    "axe X",
    "axe Y",
    "axe Z"
};
const char* NewOmniDriver::visualNodeFiles[NVISUALNODE] =
{
    "mesh/stylusO.obj",
    "mesh/articulation5O.obj",
    "mesh/articulation4O.obj",
    "mesh/articulation3O.obj",
    "mesh/articulation2O.obj",
    "mesh/articulation1O.obj",
    "mesh/BASEO.obj",
    "mesh/axeX.obj",
    "mesh/axeY.obj",
    "mesh/axeZ.obj"
};

using namespace sofa::defaulttype;

static HHD hHD = HD_INVALID_HANDLE ;
vector< HHD > hHDVector;
vector<NewOmniDriver*> autreOmniDriver;
static HDSchedulerHandle hStateHandle = HD_INVALID_HANDLE;
bool initDeviceBool;
bool frameAvant = false;
bool desktop = false;
int compteur_debug = 0;

static sofa::helper::system::atomic<int> doUpdate;



//retour en cas d'erreur
//TODO: rajouter le numero de l'interface qui pose pb
void printError(const HDErrorInfo *error, const char *message)
{
    std::cout<<hdGetErrorString(error->errorCode)<<std::endl;
    std::cout<<"HHD: "<<error->hHD<<std::endl;
    std::cout<<"Error Code: "<<error->hHD<<std::endl;
    std::cout<<"Internal Error Code: "<<error->internalErrorCode<<std::endl;
    std::cout<<"Message: "<<message<<std::endl;
}


HDCallbackCode HDCALLBACK copyDeviceDataCallback(void * userData);

//boucle qui recupere les info sur l'interface et les copie sur data->servoDeviceData
HDCallbackCode HDCALLBACK stateCallback(void * userData)
{

    if(doUpdate)
    {
        copyDeviceDataCallback(userData);
        doUpdate.dec(); // set to 0
    }

    //vector<NewOmniDriver*> autreOmniDriver = static_cast<vector<NewOmniDriver*>>(userData);
    //NewOmniData* data = static_cast<NewOmniData*>(userData);
    //FIXME : Apparenlty, this callback is run before the mechanical state initialisation. I've found no way to know whether the mechcanical state is initialized or not, so i wait ...

    RigidTypes::VecCoord positionDevs;
    RigidTypes::VecDeriv forceDevs;
    forceDevs.clear();
    positionDevs.resize(autreOmniDriver.size());
    forceDevs.resize(autreOmniDriver.size());

    for(unsigned int i=0; i<autreOmniDriver.size(); i++)
    {
        if(autreOmniDriver[i]->data.servoDeviceData.stop)
        {
            return HD_CALLBACK_DONE;
        }
        if (!autreOmniDriver[i]->data.servoDeviceData.ready)
        {
            return HD_CALLBACK_CONTINUE;
        }
        HHD hapticHD = hHDVector[i];
        hdMakeCurrentDevice(hapticHD);

        hdBeginFrame(hapticHD);

		//m_buttonState contient la valeur fusionné des boutons de l'omni. Pour recuperer ces valeurs, on passe donc par un décalage de bits.
		autreOmniDriver[i]->stateButton1 = (((autreOmniDriver[i]->data.servoDeviceData.m_buttonState)>>0)<<31)>>31;
		autreOmniDriver[i]->stateButton2 = (((autreOmniDriver[i]->data.servoDeviceData.m_buttonState)>>1)<<31)>>31;

        if((autreOmniDriver[i]->data.servoDeviceData.m_buttonState & HD_DEVICE_BUTTON_1) || autreOmniDriver[i]->data.permanent_feedback)
            hdSetDoublev(HD_CURRENT_FORCE, autreOmniDriver[i]->data.currentForce);

        autreOmniDriver[i]->data.servoDeviceData.id = hapticHD;

        // Retrieve the current button(s).
        hdGetIntegerv(HD_CURRENT_BUTTONS, &autreOmniDriver[i]->data.servoDeviceData.m_buttonState);

        //get the position
        hdGetDoublev(HD_CURRENT_POSITION, autreOmniDriver[i]->data.servoDeviceData.m_devicePosition);

        // Get the column major transform
        HDdouble transform[16];
        hdGetDoublev(HD_CURRENT_TRANSFORM, transform);

        // get Position and Rotation from transform => put in servoDeviceData
        Mat3x3d mrot;
        Quat rot;
        for (int u=0; u<3; u++)
            for (int j=0; j<3; j++)
                mrot[u][j] = transform[j*4+u];

        rot.fromMatrix(mrot);
        rot.normalize();
              
        double factor = 0.001;
        Vec3d pos(transform[12+0]*factor, transform[12+1]*factor, transform[12+2]*factor); // omni pos is in mm => sofa simulation are in meters by default
        autreOmniDriver[i]->data.servoDeviceData.pos=pos;

        
        // verify that the quaternion does not flip:
        if ((rot[0]*autreOmniDriver[i]->data.servoDeviceData.quat[0]
                +rot[1]*autreOmniDriver[i]->data.servoDeviceData.quat[1]
                +rot[2]*autreOmniDriver[i]->data.servoDeviceData.quat[2]
                +rot[3]*autreOmniDriver[i]->data.servoDeviceData.quat[3]) < 0)
            for (int u=0; u<4; u++)
                rot[u] *= -1;

        for (int u=0; u<4; u++)
            autreOmniDriver[i]->data.servoDeviceData.quat[u] = rot[u];

        //std::cout << pos << "    " << rot << std::endl;
        sofa::defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(pos* autreOmniDriver[i]->data.scale, rot);
        sofa::defaulttype::SolidTypes<double>::Transform world_H_virtualTool = autreOmniDriver[i]->data.world_H_baseOmni * baseOmni_H_endOmni * autreOmniDriver[i]->data.endOmni_H_virtualTool;
   

//partie pour ff simulatnnée
#if 1
        positionDevs[i].getCenter()=world_H_virtualTool.getOrigin();
        positionDevs[i].getOrientation()=world_H_virtualTool.getOrientation();

        //angles
        hdGetFloatv(HD_CURRENT_JOINT_ANGLES,autreOmniDriver[i]->angle1);
        hdGetFloatv(HD_CURRENT_GIMBAL_ANGLES,autreOmniDriver[i]->angle2);

        hdEndFrame(hapticHD);

    }

    for(unsigned int i=0; i<autreOmniDriver.size(); i++)
    {

		for(unsigned int j=0; j< positionDevs.size(); j++)
		{
			SReal fx, fy, fz;
			(autreOmniDriver[i]->data.forceFeedback)->computeForce(positionDevs[j].getCenter().x(),positionDevs[j].getCenter().y(), positionDevs[j].getCenter().z(), 0, 0, 0, 0, fx, fy, fz); 
			forceDevs[j] = RigidTypes::Deriv(Vec3d(fx,fy,fz), Vec3d());
		}

        /// COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
        sofa::defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni((autreOmniDriver[i]->data.servoDeviceData.pos)* autreOmniDriver[i]->data.scale, autreOmniDriver[i]->data.servoDeviceData.quat);

        Vec3d world_pos_tool = positionDevs[i].getCenter();
        Quat world_quat_tool = positionDevs[i].getOrientation();

        // we compute its value in the current Tool frame:
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_tool_inTool(world_quat_tool.inverseRotate(forceDevs[i].getVCenter()),  world_quat_tool.inverseRotate(forceDevs[i].getVOrientation())  );
        // we transport (change of application point) its value to the endOmni frame
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_endOmni_inEndOmni = autreOmniDriver[i]->data.endOmni_H_virtualTool * Wrench_tool_inTool;
        // we compute its value in the baseOmni frame
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_endOmni_inBaseOmni( baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getForce()), baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getTorque()) );

        autreOmniDriver[i]->data.currentForce[0] = Wrench_endOmni_inBaseOmni.getForce()[0] * autreOmniDriver[i]->data.forceScale;
        autreOmniDriver[i]->data.currentForce[1] = Wrench_endOmni_inBaseOmni.getForce()[1] * autreOmniDriver[i]->data.forceScale;
        autreOmniDriver[i]->data.currentForce[2] = Wrench_endOmni_inBaseOmni.getForce()[2] * autreOmniDriver[i]->data.forceScale;

        //cout<<currentForce[0]<<currentForce[1]<<currentForce[2]<<endl;

        //	if((autreOmniDriver[i]->data.servoDeviceData.m_buttonState & HD_DEVICE_BUTTON_1) || autreOmniDriver[i]->data.permanent_feedback)
        //{
        //	if(currentForce[0]>0.1)
        //		cout<<currentForce[0]<<" "<<currentForce[1]<<" "<<currentForce[2]<<endl;
        //	HHD hapticHD = hHDVector[i];
        //	hdMakeCurrentDevice(hapticHD);
        //	hdBeginFrame(hapticHD);
        //	//hdSetDoublev(HD_CURRENT_FORCE, autreOmniDriver[i]->data.currentForce);
        //	hdEndFrame(hapticHD);
        //}

        autreOmniDriver[i]->data.servoDeviceData.nupdates++;
    }

#else

        Vec3d world_pos_tool = world_H_virtualTool.getOrigin();
        Quat world_quat_tool = world_H_virtualTool.getOrientation();
        //truc sur le forcefeedback
        /////////////// 6D rendering ////////////////
        sofa::defaulttype::SolidTypes<double>::SpatialVector Twist_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0)); // Todo: compute a velocity !!
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_tool_inWorld(Vec3d(0.0,0.0,0.0), Vec3d(0.0,0.0,0.0));

        if (autreOmniDriver[i]->data.forceFeedback != NULL)
            (autreOmniDriver[i]->data.forceFeedback)->computeWrench(world_H_virtualTool,Twist_tool_inWorld,Wrench_tool_inWorld ); //en faire qu'un et uttiliser compute force

        // we compute its value in the current Tool frame:
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_tool_inTool(world_quat_tool.inverseRotate(Wrench_tool_inWorld.getForce()),  world_quat_tool.inverseRotate(Wrench_tool_inWorld.getTorque())  );
        // we transport (change of application point) its value to the endOmni frame
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_endOmni_inEndOmni = autreOmniDriver[i]->data.endOmni_H_virtualTool * Wrench_tool_inTool;
        // we compute its value in the baseOmni frame
        sofa::defaulttype::SolidTypes<double>::SpatialVector Wrench_endOmni_inBaseOmni( baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getForce()), baseOmni_H_endOmni.projectVector(Wrench_endOmni_inEndOmni.getTorque()) );

        double currentForce[3];
        currentForce[0] = Wrench_endOmni_inBaseOmni.getForce()[0] * autreOmniDriver[i]->data.forceScale;
        currentForce[1] = Wrench_endOmni_inBaseOmni.getForce()[1] * autreOmniDriver[i]->data.forceScale;
        currentForce[2] = Wrench_endOmni_inBaseOmni.getForce()[2] * autreOmniDriver[i]->data.forceScale;

        if(autreOmniDriver[i]->data.permanent_feedback)
        {
            hdSetDoublev(HD_CURRENT_FORCE, currentForce);
            HDErrorInfo error;
            if (HD_DEVICE_ERROR(error = hdGetError()))
            {
                std::cout<<hdGetErrorString(error.errorCode)<<std::endl;
                std::cout<<"HHD: "<<error.hHD<<std::endl;
                std::cout<<"Error Code: "<<error.hHD<<std::endl;
                std::cout<<"Internal Error Code: "<<error.internalErrorCode<<std::endl;
            }
        }

        autreOmniDriver[i]->data.servoDeviceData.nupdates++;

        //angles
        hdGetFloatv(HD_CURRENT_JOINT_ANGLES,autreOmniDriver[i]->angle1);
        hdGetFloatv(HD_CURRENT_GIMBAL_ANGLES,autreOmniDriver[i]->angle2);

        hdEndFrame(hapticHD);
    }

#endif


    return HD_CALLBACK_CONTINUE;
}

//stop le Scheduler
void exitHandler()
{
    hdStopScheduler();
    hdUnschedule(hStateHandle);
}


//copie les info sur le device de data->servoDeviceData a data->deviceData
//TODO: ou plutot remplir le PosD ici et gicler data->deviceData qui servirait plus a rien
HDCallbackCode HDCALLBACK copyDeviceDataCallback(void * /*pUserData*/)
{
    //NewOmniData *data = static_cast<OmniData*>(pUserData);
    //memcpy(&data->deviceData, &data->servoDeviceData, sizeof(NewDeviceData));
    //data->servoDeviceData.nupdates = 0;
    //data->servoDeviceData.ready = true;
    //vector<NewOmniDriver*> autreOmniDriver = static_cast<vector<NewOmniDriver*>>(pUserData);
    for(unsigned int i=0; i<autreOmniDriver.size(); i++)
    {
        memcpy(&autreOmniDriver[i]->data.deviceData, &autreOmniDriver[i]->data.servoDeviceData, sizeof(NewDeviceData));
        autreOmniDriver[i]->data.servoDeviceData.nupdates = 0;
        autreOmniDriver[i]->data.servoDeviceData.ready = true;
    }
    return HD_CALLBACK_DONE;
}

//stop le callback > difference avec exithandler??
HDCallbackCode HDCALLBACK stopCallback(void * /*pUserData*/)
{
    //OmniData *data = static_cast<OmniData*>(pUserData);
    //data->servoDeviceData.stop = true;
    //vector<NewOmniDriver*> autreOmniDriver = static_cast<vector<NewOmniDriver*>>(pUserData);
    for(unsigned int i=0; i<autreOmniDriver.size(); i++)
        autreOmniDriver[i]->data.servoDeviceData.stop =true;
    return HD_CALLBACK_DONE;
}

/**
 * Sets up the device,
 */
//initialise l'omni > TODO: a appeler plusieur fois depuis l'interface n�1
int NewOmniDriver::initDevice()
{
    std::cout<<"init Device is called"<<std::endl;
    HDErrorInfo error;
    for(unsigned int i=0; i<autreOmniDriver.size(); i++)
    {
        while(autreOmniDriver[i]->isInitialized && i<autreOmniDriver.size())
        {
            i++;
            if(i==autreOmniDriver.size())
                return 0;
        }

        autreOmniDriver[i]->isInitialized = true;
        autreOmniDriver[i]->data.deviceData.quat.clear();
        autreOmniDriver[i]->data.servoDeviceData.quat.clear();

        if(hHDVector[i] == HD_INVALID_HANDLE)
        {
            hHDVector[i] = hdInitDevice(autreOmniDriver[i]->deviceName.getValue().c_str());

            if (HD_DEVICE_ERROR(error = hdGetError()))
            {
              std::string m = "[NewOmni] Failed to initialize the device " + autreOmniDriver[i]->deviceName.getValue();
              printError(&error, m.c_str());
              autreOmniDriver[i]->isInitialized = false;
              return -1;
            }
            else
            {
                std::cout<<deviceName.getValue()<<"[NewOmni] Found device "<<autreOmniDriver[i]->deviceName.getValue()<<std::endl;

                hdEnable(HD_FORCE_OUTPUT);
                hdEnable(HD_MAX_FORCE_CLAMPING);
            }
        }
    }

    doUpdate = 0;
    //Start the servo loop scheduler.
    hdStartScheduler();
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        std::cout<<"[NewOmni] Failed to start the scheduler"<<std::endl;
    }

    for(unsigned int i=0; i<autreOmniDriver.size(); i++)
    {
        autreOmniDriver[i]->data.servoDeviceData.ready = false;
        autreOmniDriver[i]->data.servoDeviceData.stop = false;
    }

    hStateHandle = hdScheduleAsynchronous( stateCallback, (void*) &autreOmniDriver, HD_DEFAULT_SCHEDULER_PRIORITY);

    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        printError(&error, "erreur avec le device");
        std::cout<<deviceName.getValue()<<std::endl;
    }
    return 0;
}

//constructeur
NewOmniDriver::NewOmniDriver()
    : forceScale(initData(&forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
    , scale(initData(&scale, 100.0, "scale","Default scale applied to the Phantom Coordinates. "))
    , positionBase(initData(&positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
    , orientationBase(initData(&orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
    , positionTool(initData(&positionTool, Vec3d(0,0,0), "positionTool","Position of the tool in the omni end effector frame"))
    , orientationTool(initData(&orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool in the omni end effector frame"))
    , permanent(initData(&permanent, false, "permanent" , "Apply the force feedback permanently"))
    , omniVisu(initData(&omniVisu, false, "omniVisu", "Visualize the position of the interface in the virtual scene"))
    , posDevice(initData(&posDevice, "posDevice", "position of the base of the part of the device"))
    , posStylus(initData(&posStylus, "posStylus", "position of the base of the stylus"))
    , locDOF(initData(&locDOF,"locDOF","localisation of the DOFs MechanicalObject"))
    , deviceName(initData(&deviceName,std::string("Default PHANToM"),"deviceName","name of the device"))
    , deviceIndex(initData(&deviceIndex,1,"deviceIndex","index of the device"))
    , openTool(initData(&openTool,"openTool","opening of the tool"))
    , maxTool(initData(&maxTool,1.0,"maxTool","maxTool value"))
    , minTool(initData(&minTool,0.0,"minTool","minTool value"))
    , openSpeedTool(initData(&openSpeedTool,0.1,"openSpeedTool","openSpeedTool value"))
    , closeSpeedTool(initData(&closeSpeedTool,0.1,"closeSpeedTool","closeSpeedTool value"))
    , useScheduler(initData(&useScheduler,false,"useScheduler","Enable use of OpenHaptics Scheduler methods to synchronize haptics thread"))
    , setRestShape(initData(&setRestShape, false, "setRestShape", "True to control the rest position instead of the current position directly"))
    , applyMappings(initData(&applyMappings, true, "applyMappings", "True to enable applying the mappings after setting the position"))
    , alignOmniWithCamera(initData(&alignOmniWithCamera, true, "alignOmniWithCamera", "True to keep the Omni's movements in the same reference frame as the camera"))
	, stateButton1(initData(&stateButton1, false, "stateButton1", "True if the First button of the Omni is pressed"))
	, stateButton2(initData(&stateButton2, false, "stateButton2", "True if the Second button of the Omni is pressed"))
{
    this->f_listening.setValue(true);
    data.forceFeedback = NULL;
    noDevice = false;
    firstInit=true;
    firstDevice = true;
    addAlias(&omniVisu,"drawDevice");
}

//destructeur
NewOmniDriver::~NewOmniDriver()
{

}

//arrete le call back TODO: a ne lancer que depuis l'interface n�1
void NewOmniDriver::cleanup()
{
    std::cout << "NewOmniDriver::cleanup()" << std::endl;
    if(firstDevice)
        hdScheduleSynchronous(stopCallback, (void*) &autreOmniDriver, HD_MAX_SCHEDULER_PRIORITY);
    isInitialized = false;
}

//configure l'effort
//void NewOmniDriver::setForceFeedback(LCPForceFeedback<Rigid3dTypes>* ff)
void NewOmniDriver::setForceFeedback(ForceFeedback* ff)
{
    // the forcefeedback is already set
    if(data.forceFeedback == ff)
    {
        return;
    }

    data.forceFeedback = ff;
};

//executed once at the start of Sofa, initialization of all variables excepts haptics-related ones
void NewOmniDriver::init()
{
    sofa::simulation::Node::SPtr rootContext = static_cast<simulation::Node*>(this->getContext()->getRootContext());
    if(firstDevice)
    {
        rootContext->getTreeObjects<NewOmniDriver>(&autreOmniDriver);
        sout<<"Detected NewOmniDriver:"<<sendl;
        for(unsigned int i=0; i<autreOmniDriver.size(); i++)
        {
            sout<<"  device "<<i<<" = "<<autreOmniDriver[i]->getName()<<autreOmniDriver[i]->deviceName.getValue()<<sendl;
            autreOmniDriver[i]->deviceIndex.setValue(i);
            hHDVector.push_back(HD_INVALID_HANDLE);
            autreOmniDriver[i]->firstDevice=false;
            autreOmniDriver[i]->data.currentForce[0]=0;
            autreOmniDriver[i]->data.currentForce[1]=0;
            autreOmniDriver[i]->data.currentForce[2]=0;
        }
        firstDevice=true;
    }

    sout << deviceName.getValue()+" init" << sendl;

    if(alignOmniWithCamera.getValue())
    {
        camera = this->getContext()->get<component::visualmodel::InteractiveCamera>(this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);
        if(!camera)
        {
            camera = this->getContext()->get<component::visualmodel::InteractiveCamera>();
        }
        if (!camera)
        {
            sofa::simulation::Node::SPtr groot = dynamic_cast<simulation::Node*>(this->getContext());
            camera = sofa::core::objectmodel::New<component::visualmodel::InteractiveCamera>();
            camera->setName(core::objectmodel::Base::shortName(camera.get()));
            groot->addObject(camera);
            camera->bwdInit();
        }
        if(!camera)
        {
            serr << "Cannot find or create Camera." << sendl;
        }
    }


    modX=false;
    modY=false;
    modZ=false;
    modS=false;
    axesActif=false;

    initDeviceBool=false;

    VecCoord& posD =(*posDevice.beginEdit());
    posD.resize(NVISUALNODE+1);
    posDevice.endEdit();

    initVisu=false;
    changeScale=false;
    visuActif=false;
    isInitialized = false;
    frame=false;
    visuCreation=false;

    for(int i=0; i<NVISUALNODE; i++)
    {
        visualNode[i].visu = NULL;
        visualNode[i].mapping = NULL;
    }

    nodePrincipal = rootContext->createChild("omniVisu "+deviceName.getValue());
    nodePrincipal->updateContext();

    DOFs=NULL;

    firstInit=false;

    if(!initVisu)
    {
        rigidDOF=NULL;

        if(rigidDOF==NULL)
        {

            rigidDOF = sofa::core::objectmodel::New<MMechanicalObject>();

            nodePrincipal->addObject(rigidDOF);
            rigidDOF->name.setValue("rigidDOF");

            VecCoord& posDOF =*(rigidDOF->x.beginEdit());
            posDOF.resize(NVISUALNODE+1);
            rigidDOF->x.endEdit();

            rigidDOF->init();

            nodePrincipal->updateContext();
        }

        for(int i=0; i<NVISUALNODE; i++)
        {
            visualNode[i].node = nodePrincipal->createChild(visualNodeNames[i]);

            if(visualNode[i].visu == NULL && visualNode[i].mapping == NULL)
            {

                // create the visual model and add it to the graph //
                visualNode[i].visu = sofa::core::objectmodel::New<sofa::component::visualmodel::OglModel>();
                visualNode[i].node->addObject(visualNode[i].visu);
                visualNode[i].visu->name.setValue("VisualParticles");
                visualNode[i].visu->fileMesh.setValue(visualNodeFiles[i]);

                visualNode[i].visu->init();
                visualNode[i].visu->initVisual();
                visualNode[i].visu->updateVisual();

                // create the visual mapping and at it to the graph //
                visualNode[i].mapping = sofa::core::objectmodel::New< sofa::component::mapping::RigidMapping< Rigid3dTypes, ExtVec3fTypes > > ();
                visualNode[i].node->addObject(visualNode[i].mapping);
                visualNode[i].mapping->setModels(rigidDOF.get(), visualNode[i].visu.get());
                visualNode[i].mapping->name.setValue("RigidMapping");
                visualNode[i].mapping->f_mapConstraints.setValue(false);
                visualNode[i].mapping->f_mapForces.setValue(false);
                visualNode[i].mapping->f_mapMasses.setValue(false);
                //visualNode[i].mapping->m_inputObject.setValue("@../RigidDOF");
                //visualNode[i].mapping->m_outputObject.setValue("@VisualParticles");
                visualNode[i].mapping->index.setValue(i+1);
                visualNode[i].mapping->init();
            }
            if(i<VN_X)
                nodePrincipal->removeChild(visualNode[i].node);
        }

        visualNode[VN_X].visu->setColor(1.0,0.0,0.0,0);
        visualNode[VN_Y].visu->setColor(0.0,1.0,0.0,0);
        visualNode[VN_Z].visu->setColor(0.0,0.0,1.0,0);

        nodePrincipal->updateContext();

        for(int i=0; i<NVISUALNODE; i++)
        {
            visualNode[i].node->updateContext();
        }

        for(int j=0; j<=VN_X; j++)
        {
            sofa::defaulttype::ResizableExtVector< sofa::defaulttype::Vec<3,float> > &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
            for(unsigned int i=0; i<scaleMapping.size(); i++)
                scaleMapping[i] *= (float)(1.0*scale.getValue()/100.0);
            visualNode[j].mapping->points.endEdit();
        }

        oldScale=(float)scale.getValue();
        changeScale=false;
        initVisu=true;
        visuActif=false;
    }

    Vec1d& openT = (*openTool.beginEdit());
    openT[0]=maxTool.getValue();
    openTool.endEdit();


}


//recupere dans la scene l'effort a donner a l'interface
void NewOmniDriver::bwdInit()
{
    sout<<"NewOmniDriver::bwdInit()"<<sendl;

    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
	ForceFeedback* ff = context->get<ForceFeedback>(this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);
    if(ff)
        this->setForceFeedback(ff);
	else
	{
		serr<< "Warning(NewOmniDriver): No ForceFeedback found." << sendl;
	}

    setDataValue();

    if(firstDevice && initDevice()==-1)
    {
        noDevice=true;
        serr<<"NO DEVICE"<<sendl;
    }

		DOFs = context->get<sofa::component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> > (this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);

		if (DOFs==NULL)
        {
            serr<<" no MechanicalObject with template = Rigid found"<<sendl;
        }
        else
        {
          sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<VecCoord> > xfree = *DOFs->write(this->setRestShape.getValue() ? sofa::core::VecCoordId::restPosition() : sofa::core::VecCoordId::freePosition());
          if (xfree.size() == 0)
            xfree.resize(1);
          autreOmniDriver[this->deviceIndex.getValue()]->DOFs = DOFs;
        }
}

//configure data
void NewOmniDriver::setDataValue()
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

//lance toute les fonction de reset (cas d'un update)
void NewOmniDriver::reset()
{
    std::cout<<"NewOmniDriver::reset() is called" <<std::endl;
    this->reinit();
}

//idem
void NewOmniDriver::reinit()
{
    std::cout<<"NewOmniDriver::reinit() is called" <<std::endl;

    this->cleanup();
    this->bwdInit();
    if(data.scale!=oldScale)
        changeScale = true;

    std::cout<<"NewOmniDriver::reinit() done" <<std::endl;
}

void NewOmniDriver::draw(const core::visual::VisualParams* vparam){
	draw();
}

// setup omni device visualization
void NewOmniDriver::draw()
{
    if(initVisu)
    {
        if(!visuActif && omniVisu.getValue())
        {
            for(int i=0; i<VN_X; i++)
            {
                nodePrincipal->addChild(visualNode[i].node);
                visualNode[i].node->updateContext();
            }
            nodePrincipal->updateContext();
            visuActif=true;
        }
        VecCoord& posD =(*posDevice.beginEdit());
        VecCoord& posDOF =*(rigidDOF->x.beginEdit());
        posD.resize(NVISUALNODE+1);
        posDOF.resize(NVISUALNODE+1);
        for(int i=0; i<NVISUALNODE+1; i++)
        {
            posDOF[i].getCenter() = posD[i].getCenter();
            posDOF[i].getOrientation() = posD[i].getOrientation();
        }
        //for(int i=0;i<NVISUALNODE;i++)
        //{
        //	if(omniVisu.getValue() || i>6)
        //	{
        //		visualNode[i].visu->drawVisual();
        //		visualNode[i].mapping->draw();
        //	}
        //}
        rigidDOF->x.endEdit();
        posDevice.endEdit();


        //scale
        if(changeScale)
        {
            float rapport=((float)data.scale)/oldScale;
            for(int j = 0; j<NVISUALNODE ; j++)
            {
                sofa::defaulttype::ResizableExtVector< sofa::defaulttype::Vec<3,float> > &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
                for(unsigned int i=0; i<scaleMapping.size(); i++)
                    scaleMapping[i]*=rapport;
                visualNode[j].mapping->points.endEdit();
                oldScale=(float)data.scale;
            }
            changeScale=false;
        }
    }
    //delete omnivisual
    if(initVisu && visuActif && !omniVisu.getValue())
    {
        for(int i=0; i<VN_X; i++)
        {
            nodePrincipal->removeChild(visualNode[i].node);
        }
        visuActif=false;
    }

}

void NewOmniDriver::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kpe)
{
    //cout<<kpe->getKey()<<" "<<int(kpe->getKey())<<std::endl;
    if(axesActif && omniVisu.getValue())
    {
        if ((kpe->getKey()=='X' || kpe->getKey()=='x') && !modX )
        {
            modX=true;
        }
        if ((kpe->getKey()=='Y' || kpe->getKey()=='y') && !modY )
        {
            modY=true;
        }
        if ((kpe->getKey()=='Z' || kpe->getKey()=='z') && !modZ )
        {
            modZ=true;
        }
        if ((kpe->getKey()=='Q' || kpe->getKey()=='q') && !modS )
        {
            modS=true;
        }
        if (kpe->getKey()==18) //left
        {
            if(modX || modY || modZ)
            {
                Quat& orientB =(*orientationBase.beginEdit());
                Vec3d deplacement=orientB.rotate(Vec3d(-(int)modX,-(int)modY,-(int)modZ));
                orientationBase.endEdit();
                Vec3d& posB =(*positionBase.beginEdit());
                posB+=deplacement;
                positionBase.endEdit();
            }
            else if(modS)
            {
                data.scale--;
                changeScale = true;
            }
        }
        else if (kpe->getKey()==20) //right
        {

            if(modX || modY || modZ)
            {
                Quat& orientB =(*orientationBase.beginEdit());
                Vec3d deplacement=orientB.rotate(Vec3d((int)modX,(int)modY,(int)modZ));
                orientationBase.endEdit();
                Vec3d& posB =(*positionBase.beginEdit());
                posB+=deplacement;
                positionBase.endEdit();
            }
            else if(modS)
            {
                data.scale++;
                changeScale = true;
            }
        }
        else if ((kpe->getKey()==21) && (modX || modY || modZ)) //down
        {
            Quat& orientB =(*orientationBase.beginEdit());
            sofa::helper::Quater<double> quarter_transform(Vec3d((int)modX,(int)modY,(int)modZ),-M_PI/50);
            orientB*=quarter_transform;
            orientationBase.endEdit();
        }
        else if ((kpe->getKey()==19) && (modX || modY || modZ)) //up
        {
            Quat& orientB =(*orientationBase.beginEdit());
            sofa::helper::Quater<double> quarter_transform(Vec3d((int)modX,(int)modY,(int)modZ),M_PI/50);
            orientB*=quarter_transform;
            orientationBase.endEdit();
        }
        if ((kpe->getKey()=='E' || kpe->getKey()=='e'))
        {
            std::cout<<"reset position"<<std::endl;

            Quat& orientB =(*orientationBase.beginEdit());
            orientB.clear();
            orientationBase.endEdit();

            Vec3d& posB =(*positionBase.beginEdit());
            posB.clear();
            positionBase.endEdit();
        }
    }
    if ((kpe->getKey()==48+deviceIndex.getValue()) && initVisu)
    {
        if(!axesActif)
        {
            visualNode[VN_X].visu->setColor(1.0,0.0,0.0,1);
            visualNode[VN_Y].visu->setColor(0.0,1.0,0.0,1);
            visualNode[VN_Z].visu->setColor(0.0,0.0,1.0,1);
            axesActif=true;
        }
        else
        {
            visualNode[VN_X].visu->setColor(1.0,0.0,0.0,0);
            visualNode[VN_Y].visu->setColor(0.0,1.0,0.0,0);
            visualNode[VN_Z].visu->setColor(0.0,0.0,1.0,0);
            axesActif=false;
        }
    }
}

void NewOmniDriver::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kre)
{
    if (kre->getKey()=='X' || kre->getKey()=='x' )
    {
        modX=false;
    }
    if (kre->getKey()=='Y' || kre->getKey()=='y' )
    {
        modY=false;
    }
    if (kre->getKey()=='Z' || kre->getKey()=='z' )
    {
        modZ=false;
    }
    if (kre->getKey()=='Q' || kre->getKey()=='q' )
    {
        modS=false;
    }
}


//boucle animation
void NewOmniDriver::onAnimateBeginEvent()
{
    // copy data->servoDeviceData to gDeviceData
	if (useScheduler.getValue())
		hdScheduleSynchronous(copyDeviceDataCallback, (void*) &autreOmniDriver, HD_MAX_SCHEDULER_PRIORITY);
	else
	{
		doUpdate.inc(); // set to 1
		while(doUpdate)
		{
#ifdef SOFA_HAVE_BOOST
			boost::thread::yield();
#else
			sofa::helper::system::thread::CTime::sleep(0);
#endif
		}
	}
    if (data.deviceData.ready)
    {
        data.deviceData.quat.normalize();

        // COMPUTATION OF THE vituralTool 6D POSITION IN THE World COORDINATES
        sofa::defaulttype::SolidTypes<double>::Transform baseOmni_H_endOmni(data.deviceData.pos*data.scale, data.deviceData.quat);

    
        Quat& orientB =(*orientationBase.beginEdit());
        Vec3d& posB =(*positionBase.beginEdit());
        if(alignOmniWithCamera.getValue())
        {
            Quat cameraRotation = camera->getOrientation();
            orientB = cameraRotation;
        }
        orientB.normalize();
        data.world_H_baseOmni.set(posB,orientB);
        orientationBase.endEdit();
        positionBase.endEdit();

        VecCoord& posD =(*posDevice.beginEdit());
        //posD.resize(NVISUALNODE+1);

        sofa::defaulttype::SolidTypes<double>::Transform world_H_virtualTool = data.world_H_baseOmni * baseOmni_H_endOmni * data.endOmni_H_virtualTool;
        sofa::defaulttype::SolidTypes<double>::Transform tampon = data.world_H_baseOmni;

        sofa::helper::Quater<double> q;
#if 1
        //get position base
        posD[0].getCenter() =  tampon.getOrigin();
        posD[0].getOrientation() =  tampon.getOrientation();

        //get position stylus
        tampon*=baseOmni_H_endOmni;
        posD[1+VN_stylus] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos joint 2
        sofa::helper::Quater<double> quarter2(Vec3d(0.0,0.0,1.0),angle2[2]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr2(Vec3d(0.0,0.0,0.0),quarter2);
        tampon*=transform_segr2;
        posD[1+VN_joint2] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos joint 1
        sofa::helper::Quater<double> quarter3(Vec3d(1.0,0.0,0.0),angle2[1]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr3(Vec3d(0.0,0.0,0.0),quarter3);
        tampon*=transform_segr3;
        posD[1+VN_joint1] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos arm 2
        sofa::helper::Quater<double> quarter4(Vec3d(0.0,1.0,0.0),-angle2[0]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr4(Vec3d(0.0,0.0,0.0),quarter4);
        tampon*=transform_segr4;
        posD[1+VN_arm2] = Coord(tampon.getOrigin(), tampon.getOrientation());
        //get pos arm 1
        sofa::helper::Quater<double> quarter5(Vec3d(1.0,0.0,0.0),-(M_PI/2)+angle1[2]-angle1[1]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr5(Vec3d(0.0,13.33*data.scale/100,0.0),quarter5);
        tampon*=transform_segr5;
        posD[1+VN_arm1] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos joint 0
        sofa::helper::Quater<double> quarter6(Vec3d(1.0,0.0,0.0),angle1[1]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr6(Vec3d(0.0,13.33*data.scale/100,0.0),quarter6);
        tampon*=transform_segr6;
        posD[1+VN_joint0] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos base
        sofa::helper::Quater<double> quarter7(Vec3d(0.0,0.0,1.0),angle1[0]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr7(Vec3d(0.0,0.0,0.0),quarter7);
        tampon*=transform_segr7;
        posD[1+VN_base] = Coord(tampon.getOrigin(), tampon.getOrientation());
#else
        q.clear();
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr[6];
        transform_segr[0].set(Vec3d(0.0,0.0,0.0),q);//get position base
        transform_segr[1].set(baseOmni_H_endOmni.getOrigin(),baseOmni_H_endOmni.getOrientation());//get position stylus
        transform_segr[2].set(Vec3d(0.0,0.0,0.0),q.axisToQuat(Vec3d(0.0,0.0,1.0),angle2[2]));//get pos articulation 2
        transform_segr[3].set(Vec3d(0.0,0.0,0.0),q.axisToQuat(Vec3d(1.0,0.0,0.0),angle2[1]));//get pos articulation 1
        transform_segr[4].set(Vec3d(0.0,0.0,0.0),q.axisToQuat(Vec3d(0.0,1.0,0.0),-angle2[0]));//get pos arm 2
        transform_segr[5].set(Vec3d(0.0,13.33*data.scale/100,0.0),q.axisToQuat(Vec3d(1.0,0.0,0.0),-(float)(pi/2)+angle1[2]-angle1[1]));//get pos arm 1
        transform_segr[6].set(Vec3d(0.0,13.33*data.scale/100,0.0),q.axisToQuat(Vec3d(1.0,0.0,0.0),angle1[1]));//get pos articulation 0
        transform_segr[7].set(Vec3d(0.0,0.0,0.0),q.axisToQuat(Vec3d(0.0,0.0,1.0),angle1[0]));//get pos base

        for(int i=0; i<8; i++)
        {
            tampon*=transform_segr[i];
            posD[i].getCenter() =  tampon.getOrigin();
            posD[i].getOrientation() =  tampon.getOrientation();
        }
#endif
        //get pos of axes

        posD[1+VN_X].getCenter() =  data.world_H_baseOmni.getOrigin();
        posD[1+VN_Y].getCenter() =  data.world_H_baseOmni.getOrigin();
        posD[1+VN_Z].getCenter() =  data.world_H_baseOmni.getOrigin();
        posD[1+VN_X].getOrientation() =  (data.world_H_baseOmni).getOrientation()*q.axisToQuat(Vec3d(0.0,0.0,1.0),-M_PI/2);
        posD[1+VN_Y].getOrientation() =  (data.world_H_baseOmni).getOrientation()*q.axisToQuat(Vec3d(1.0,0.0,0.0),0);
        posD[1+VN_Z].getOrientation() = (data.world_H_baseOmni).getOrientation()*q.axisToQuat(Vec3d(1.0,0.0,0.0),-M_PI/2);

        posDevice.endEdit();

        if(DOFs!=NULL)
        {
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<VecCoord> > x = *DOFs->write(this->setRestShape.getValue() ? sofa::core::VecCoordId::restPosition() : sofa::core::VecCoordId::position());
            sofa::helper::WriteAccessor<sofa::core::objectmodel::Data<VecCoord> > xfree = *DOFs->write(this->setRestShape.getValue() ? sofa::core::VecCoordId::restPosition() : sofa::core::VecCoordId::freePosition());
            unsigned int index = 0;

            x    [index].getCenter()=world_H_virtualTool.getOrigin();
            xfree[index].getCenter()=world_H_virtualTool.getOrigin();
            x    [index].getOrientation()=world_H_virtualTool.getOrientation();
            xfree[index].getOrientation()=world_H_virtualTool.getOrientation();
        }
        if (applyMappings.getValue())
        {
            sofa::simulation::Node *node = dynamic_cast<sofa::simulation::Node*> (this->getContext());
            if (node)
            {
                sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(node);
                sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(node);
            }
        }
        //button state
        Vec1d& openT = (*openTool.beginEdit());
        if(data.deviceData.m_buttonState & HD_DEVICE_BUTTON_1)
        {
            if(openT[0]>minTool.getValue())
                openT[0]-=closeSpeedTool.getValue();
            else
                openT[0]=minTool.getValue();
        }
        else
        {
            if(openT[0]<maxTool.getValue())
                openT[0]+=openSpeedTool.getValue();
            else
                openT[0]=maxTool.getValue();
        }
        openTool.endEdit();

        if( data.forceFeedback)
        {
            // store actual position of interface for the forcefeedback (as it will be used as soon as new LCP will be computed)
            data.forceFeedback->setReferencePosition(world_H_virtualTool);
        }
        /// TODO : SHOULD INCLUDE VELOCITY !!
    }
    else
        std::cout<<"data not ready \n"<<std::endl;
}

//boucle qui se declenche si il y a un evenement
void NewOmniDriver::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        onAnimateBeginEvent();
    }
    else if (dynamic_cast<core::objectmodel::KeypressedEvent *>(event))
    {
        core::objectmodel::KeypressedEvent *kpe = dynamic_cast<core::objectmodel::KeypressedEvent *>(event);
        onKeyPressedEvent(kpe);
    }
    else if (dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event))
    {
        core::objectmodel::KeyreleasedEvent *kre = dynamic_cast<core::objectmodel::KeyreleasedEvent *>(event);
        onKeyReleasedEvent(kre);
    }
}

int NewOmniDriverClass = core::RegisterObject("Solver to test compliance computation for new articulated system objects")
        .add< NewOmniDriver >()
        .addAlias("DefaultHapticsDevice")
        ;

SOFA_DECL_CLASS(NewOmniDriver)

} // namespace controller

} // namespace component

} // namespace sofa
