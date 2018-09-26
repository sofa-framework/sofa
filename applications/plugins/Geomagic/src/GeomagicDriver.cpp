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

#include "GeomagicDriver.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/core/objectmodel/ScriptEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/core/visual/VisualParams.h>


namespace sofa
{

namespace component
{

namespace controller
{


const char* GeomagicDriver::visualNodeNames[NVISUALNODE] =
{
    "stylus",
    "joint 2",
    "joint 1",
    "arm 2",
    "arm 1",
    "joint 0",
    "base"
};
const char* GeomagicDriver::visualNodeFiles[NVISUALNODE] =
{
    "mesh/stylusO.obj",
    "mesh/articulation5O.obj",
    "mesh/articulation4O.obj",
    "mesh/articulation3O.obj",
    "mesh/articulation2O.obj",
    "mesh/articulation1O.obj",
    "mesh/BASEO.obj"
};


static Mat<4,4, GLdouble> getInitialTransform() {
    Mat<4,4, GLdouble> M;

    const sofa::defaulttype::SolidTypes<double>::Transform transformOmni1(Vector3(0,0,0),Quat(Vector3(1,0,0),-M_PI/2));
    const sofa::defaulttype::SolidTypes<double>::Transform transformOmni2(Vector3(0,0,2.3483),Quat(Vector3(0,0,1),-M_PI/2));
    const sofa::defaulttype::SolidTypes<double>::Transform transformOmni3(Vector3(-16.8473,0,0),Quat(Vector3(1,0,0),M_PI));

    sofa::defaulttype::SolidTypes<double>::Transform T = transformOmni1*transformOmni2*transformOmni3;
    T.writeOpenGlMatrix((double*) M.ptr());

    return M;
}

//change the axis of the omni
static const Mat<4,4, GLdouble> initialTransform = getInitialTransform();

using namespace sofa::defaulttype;

void printError(const HDErrorInfo *error, const char *message)
{
    msg_error("GeomagicDriver::HD_DEVICE_ERROR") << hdGetErrorString(error->errorCode);
    msg_error("GeomagicDriver::HD_DEVICE_ERROR") << "HHD: "<< error->hHD;
    msg_error("GeomagicDriver::HD_DEVICE_ERROR") << "Error Code: "<< error->hHD;
    msg_error("GeomagicDriver::HD_DEVICE_ERROR") << "Internal Error Code: "<< error->internalErrorCode;
    msg_error("GeomagicDriver::HD_DEVICE_ERROR") << "Message: " << message;
}

HDCallbackCode HDCALLBACK copyDeviceDataCallback(void * pUserData)
{
    GeomagicDriver * driver = (GeomagicDriver * ) pUserData;
    driver->m_simuData = driver->m_omniData;
        return HD_CALLBACK_CONTINUE;
}

HDCallbackCode HDCALLBACK stateCallback(void * userData)
{
    HDErrorInfo error;
    GeomagicDriver * driver = (GeomagicDriver * ) userData;

    if (!driver->m_simulationStarted)
        return HD_CALLBACK_CONTINUE;

    hdMakeCurrentDevice(driver->m_hHD);
    if (HD_DEVICE_ERROR(error = hdGetError())) return HD_CALLBACK_CONTINUE;

    hdBeginFrame(driver->m_hHD);
    if (HD_DEVICE_ERROR(error = hdGetError())) return HD_CALLBACK_CONTINUE;

    hdGetIntegerv(HD_CURRENT_BUTTONS, &driver->m_omniData.buttonState);
    hdGetDoublev(HD_CURRENT_TRANSFORM, driver->m_omniData.transform);

    //angles
    hdGetDoublev(HD_CURRENT_JOINT_ANGLES,driver->m_omniData.angle1);
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES,driver->m_omniData.angle2);

    Vector3 currentForce;

    if (driver->m_forceFeedback)
    {
        Vector3 pos(driver->m_omniData.transform[12+0]*0.1,driver->m_omniData.transform[12+1]*0.1,driver->m_omniData.transform[12+2]*0.1);
        Vector3 pos_in_world = driver->d_positionBase.getValue() + driver->d_orientationBase.getValue().rotate(pos*driver->d_scale.getValue());

        driver->m_forceFeedback->computeForce(pos_in_world[0],pos_in_world[1],pos_in_world[2], 0, 0, 0, 0, currentForce[0], currentForce[1], currentForce[2]);
    }
    else
    {
        Vector3 inputForceFeedback = driver->d_inputForceFeedback.getValue();
        double normValue = inputForceFeedback.norm();
        double maxInputForceFeedback = driver->d_maxInputForceFeedback.getValue();

        if( maxInputForceFeedback > 0.0)
        {
            if( normValue > maxInputForceFeedback )
            {
                msg_warning(driver) << "Force given to applied inputForceFeedback (norm = "<< normValue <<") exceeds the maxInputForceFeedback ("<< maxInputForceFeedback <<")";

                inputForceFeedback[0] *= maxInputForceFeedback/normValue;
                inputForceFeedback[1] *= maxInputForceFeedback/normValue;
                inputForceFeedback[2] *= maxInputForceFeedback/normValue;

                driver->d_inputForceFeedback.setValue(inputForceFeedback);
            }
            currentForce = driver->d_inputForceFeedback.getValue();
        }
        else
        {
            msg_error(driver) << "maxInputForceFeedback value ("<< maxInputForceFeedback <<") is negative or 0, it should be strictly positive";
        }
    }

    Vector3 force_in_omni = driver->d_orientationBase.getValue().inverseRotate(currentForce)  * driver->d_forceScale.getValue();

    double omni_force[3];
    omni_force[0] = force_in_omni[0];
    omni_force[1] = force_in_omni[1];
    omni_force[2] = force_in_omni[2];

    hdSetDoublev(HD_CURRENT_FORCE, omni_force);

    hdEndFrame(driver->m_hHD);

    return HD_CALLBACK_CONTINUE;
}

//constructeur
GeomagicDriver::GeomagicDriver()
    : d_deviceName(initData(&d_deviceName, std::string("Default Device"), "deviceName","Name of device Configuration"))
    , d_positionBase(initData(&d_positionBase, Vec3d(0,0,0), "positionBase","Position of the interface base in the scene world coordinates"))
    , d_orientationBase(initData(&d_orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the interface base in the scene world coordinates"))
    , d_orientationTool(initData(&d_orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool"))
    , d_dh_theta(initData(&d_dh_theta, Vec6d(0,0,0,M_PI/2,M_PI/2,M_PI/2), "dh_teta","Denavit theta"))
    , d_dh_alpha(initData(&d_dh_alpha, Vec6d(0,-M_PI/2,0,M_PI/2,-M_PI/2,M_PI/2), "dh_alpha","Denavit alpha"))
    , d_dh_d(initData(&d_dh_d, Vec6d(0,0,0,0,13.3333,0), "dh_d","Denavit d"))
    , d_dh_a(initData(&d_dh_a, Vec6d(0,0,13.3333,0,0,0), "dh_a","Denavit a"))
    , d_angle(initData(&d_angle, "angle","Angluar values of joint (rad)"))
    , d_scale(initData(&d_scale, 1.0, "scale","Default scale applied to the Phantom Coordinates"))
    , d_forceScale(initData(&d_forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
    , d_frameVisu(initData(&d_frameVisu, false, "drawDeviceFrame", "Visualize the frame corresponding to the device tooltip"))
    , d_omniVisu(initData(&d_omniVisu, false, "drawDevice", "Visualize the Geomagic device in the virtual scene"))
    , d_posDevice(initData(&d_posDevice, "positionDevice", "position of the base of the part of the device"))
    , d_button_1(initData(&d_button_1,"button1","Button state 1"))
    , d_button_2(initData(&d_button_2,"button2","Button state 2"))
    , d_emitButtonEvent(initData(&d_emitButtonEvent, false, "emitButtonEvent", "If true, will send event through the graph when button are pushed/released"))
    , d_inputForceFeedback(initData(&d_inputForceFeedback, Vec3d(0,0,0), "inputForceFeedback","Input force feedback in case of no LCPForceFeedback is found (manual setting)"))
    , d_maxInputForceFeedback(initData(&d_maxInputForceFeedback, double(1.0), "maxInputForceFeedback","Maximum value of the normed input force feedback for device security"))
    , m_simulationStarted(false)
    , m_errorDevice(0)
{
    this->f_listening.setValue(true);
    m_forceFeedback = NULL;
}

GeomagicDriver::~GeomagicDriver()
{
    hdMakeCurrentDevice(m_hHD);

    hdStopScheduler();

    for (std::vector< HDCallbackCode >::iterator i = m_hStateHandles.begin();
            i != m_hStateHandles.end(); ++i)
    {
            hdUnschedule(*i);
    }
    m_hStateHandles.clear();

    hdDisableDevice(m_hHD);
}

//executed once at the start of Sofa, initialization of all variables excepts haptics-related ones
void GeomagicDriver::init()
{
    m_initVisuDone = false;
    m_errorDevice = 0;
    HDErrorInfo error;

    HDSchedulerHandle hStateHandle = HD_INVALID_HANDLE;
    m_hHD = hdInitDevice(d_deviceName.getValue().c_str());

    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        msg_error() << "Failed to initialize the device called " << d_deviceName.getValue().c_str();
        d_omniVisu.setValue(false);
        m_errorDevice = error.errorCode;
        //init the positionDevice data to avoid any crash in the scene
        m_posDeviceVisu.clear();
        m_posDeviceVisu.resize(1);
        return;
    }

    hdMakeCurrentDevice(m_hHD);
    hdEnable(HD_FORCE_OUTPUT);

    hStateHandle = hdScheduleAsynchronous(stateCallback, this, HD_MAX_SCHEDULER_PRIORITY);
    m_hStateHandles.push_back(hStateHandle);

    hStateHandle = hdScheduleAsynchronous(copyDeviceDataCallback, this, HD_MIN_SCHEDULER_PRIORITY);
    m_hStateHandles.push_back(hStateHandle);

    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        printError(&error, "Error with the device Default PHANToM");
        m_errorDevice = true;
        return;
    }

    if(d_maxInputForceFeedback.isSet())
    {
        msg_info() << "maxInputForceFeedback value ("<< d_maxInputForceFeedback.getValue() <<") is set, carefully set the max force regarding your haptic device";

        if(d_maxInputForceFeedback.getValue() <= 0.0)
        {
            msg_error() << "maxInputForceFeedback value ("<< d_maxInputForceFeedback.getValue() <<") is negative or 0, it should be strictly positive";
            d_maxInputForceFeedback.setValue(0.0);
        }
    }

    reinit();

    //Initialization of the visual components
    //resize vectors
    m_posDeviceVisu.resize(NVISUALNODE+1);

    m_visuActive = false;

    for(int i=0; i<NVISUALNODE; i++)
    {
        visualNode[i].visu = NULL;
        visualNode[i].mapping = NULL;
    }

    //create a specific node containing rigid position for visual models
    sofa::simulation::Node::SPtr rootContext = static_cast<simulation::Node*>(this->getContext()->getRootContext());
    m_omniVisualNode = rootContext->createChild("omniVisu "+d_deviceName.getValue());
    m_omniVisualNode->updateContext();

    rigidDOF = sofa::core::objectmodel::New<component::container::MechanicalObject<sofa::defaulttype::Rigid3dTypes> >();
    m_omniVisualNode->addObject(rigidDOF);
    rigidDOF->name.setValue("rigidDOF");

    VecCoord& posDOF =*(rigidDOF->x.beginEdit());
    posDOF.resize(NVISUALNODE+1);
    rigidDOF->x.endEdit();

    rigidDOF->init();
    m_omniVisualNode->updateContext();


    //creation of subnodes for each part of the device visualization
    for(int i=0; i<NVISUALNODE; i++)
    {
        visualNode[i].node = m_omniVisualNode->createChild(visualNodeNames[i]);

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
            visualNode[i].mapping->index.setValue(i+1);
            visualNode[i].mapping->init();
        }
        if(i<NVISUALNODE)
            m_omniVisualNode->removeChild(visualNode[i].node);
    }

    m_omniVisualNode->updateContext();

    for(int i=0; i<NVISUALNODE; i++)
    {
        visualNode[i].node->updateContext();
    }

    m_initVisuDone = true;

    for(int j=0; j<NVISUALNODE; j++)
    {
        sofa::defaulttype::ResizableExtVector< sofa::defaulttype::Vec<3,float> > &scaleMapping = *(visualNode[j].mapping->points.beginEdit());
        for(size_t i=0; i<scaleMapping.size(); i++)
            scaleMapping[i] *= (float)(d_scale.getValue());
        visualNode[j].mapping->points.endEdit();
    }
}

void GeomagicDriver::bwdInit()
{
    if(m_errorDevice)
        return;

    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
    m_forceFeedback = context->get<ForceFeedback>(this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);


    hdStartScheduler();
    HDErrorInfo error;

    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        msg_info() <<"Failed to start the scheduler";
        m_errorDevice = true;
        if (m_hStateHandles.size()) m_hStateHandles[0] = HD_INVALID_HANDLE;
            return;
    }
    updatePosition();
}

Mat<4,4, GLdouble> GeomagicDriver::compute_dh_Matrix(double teta,double alpha, double a, double d)
{
    Mat<4,4, GLdouble> M;

    const double dh_ct = cos(teta);
    const double dh_st = sin(teta);
    const double dh_ca = cos(alpha);
    const double dh_sa = sin(alpha);
    const double dh_a = a;
    const double dh_d = d;

    M[0][0] = dh_ct;
    M[0][1] = -dh_st*dh_ca;
    M[0][2] = dh_st*dh_sa;
    M[0][3] = dh_a*dh_ct;

    M[1][0] = dh_st;
    M[1][1] = dh_ct*dh_ca;
    M[1][2] = -dh_ct*dh_sa;
    M[1][3] = dh_a*dh_st;

    M[2][0] = 0;
    M[2][1] = dh_sa;
    M[2][2] = dh_ca;
    M[2][3] = dh_d;

    M[3][0] = 0;
    M[3][1] = 0;
    M[3][2] = 0;
    M[3][3] = 1;

    return M;
}


void GeomagicDriver::reinit()
{
    if(m_errorDevice != 0)
    {
        Quat * q_b = d_orientationBase.beginEdit();
        q_b->normalize();
        d_orientationBase.endEdit();

        Quat * q_t = d_orientationTool.beginEdit();
        q_t->normalize();
        d_orientationTool.endEdit();

        for (int i=0;i<NBJOINT;i++) 
            m_dh_matrices[i] = compute_dh_Matrix(d_dh_theta.getValue()[i],d_dh_alpha.getValue()[i],d_dh_a.getValue()[i],d_dh_d.getValue()[i]);
    }
}

void GeomagicDriver::updatePosition()
{
    Mat3x3d mrot;

    Vector6 & angle = *d_angle.beginEdit();
    GeomagicDriver::Coord & posDevice = *d_posDevice.beginEdit();    

    const Vector3 & positionBase = d_positionBase.getValue();
    const Quat & orientationBase = d_orientationBase.getValue();
    const Quat & orientationTool = d_orientationTool.getValue();
    const double & scale = d_scale.getValue();

    // update button state
    updateButtonStates(d_emitButtonEvent.getValue());

    //copy angle
    angle[0] = m_simuData.angle1[0];
    angle[1] = m_simuData.angle1[1];
    angle[2] = -(M_PI/2)+m_simuData.angle1[2]-m_simuData.angle1[1];
    angle[3] = -(M_PI/2)-m_simuData.angle2[0];
    angle[4] = m_simuData.angle2[1];
    angle[5] = -(M_PI/2)-m_simuData.angle2[2];

    //copy the position of the tool
    Vector3 position;
    position[0] = m_simuData.transform[12+0] * 0.1;
    position[1] = m_simuData.transform[12+1] * 0.1;
    position[2] = m_simuData.transform[12+2] * 0.1;

    //copy rotation of the tool
    Quat orientation;
    for (int u=0; u<3; u++)
        for (int j=0; j<3; j++)
            mrot[u][j] = m_simuData.transform[j*4+u];
    orientation.fromMatrix(mrot);

    //compute the position of the tool (according to positionbase, orientation base and the scale
    posDevice.getCenter() = positionBase + orientationBase.rotate(position*scale);
    posDevice.getOrientation() = orientationBase * orientation * orientationTool;


    if(m_initVisuDone && d_omniVisu.getValue())
    {
        sofa::defaulttype::SolidTypes<double>::Transform tampon;
        m_posDeviceVisu[0] = posDevice;
        tampon.set(m_posDeviceVisu[0].getCenter(), m_posDeviceVisu[0].getOrientation());

        //get position stylus
        m_posDeviceVisu[1+VN_stylus] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos joint 2
        sofa::helper::Quater<double> quarter2(Vec3d(0.0,0.0,1.0),m_simuData.angle2[2]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr2(Vec3d(0.0,0.0,0.0),quarter2);
        tampon*=transform_segr2;
        m_posDeviceVisu[1+VN_joint2] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos joint 1
        sofa::helper::Quater<double> quarter3(Vec3d(1.0,0.0,0.0),m_simuData.angle2[1]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr3(Vec3d(0.0,0.0,0.0),quarter3);
        tampon*=transform_segr3;
        m_posDeviceVisu[1+VN_joint1] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos arm 2
        sofa::helper::Quater<double> quarter4(Vec3d(0.0,1.0,0.0),-m_simuData.angle2[0]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr4(Vec3d(0.0,0.0,0.0),quarter4);
        tampon*=transform_segr4;
        m_posDeviceVisu[1+VN_arm2] = Coord(tampon.getOrigin(), tampon.getOrientation());
        //get pos arm 1
        sofa::helper::Quater<double> quarter5(Vec3d(1.0,0.0,0.0),-(M_PI/2)+m_simuData.angle1[2]-m_simuData.angle1[1]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr5(Vec3d(0.0,13.33*d_scale.getValue(),0.0),quarter5);
        tampon*=transform_segr5;
        m_posDeviceVisu[1+VN_arm1] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos joint 0
        sofa::helper::Quater<double> quarter6(Vec3d(1.0,0.0,0.0),m_simuData.angle1[1]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr6(Vec3d(0.0,13.33*d_scale.getValue(),0.0),quarter6);
        tampon*=transform_segr6;
        m_posDeviceVisu[1+VN_joint0] = Coord(tampon.getOrigin(), tampon.getOrientation());

        //get pos base
        sofa::helper::Quater<double> quarter7(Vec3d(0.0,0.0,1.0),m_simuData.angle1[0]);
        sofa::defaulttype::SolidTypes<double>::Transform transform_segr7(Vec3d(0.0,0.0,0.0),quarter7);
        tampon*=transform_segr7;
        m_posDeviceVisu[1+VN_base] = Coord(tampon.getOrigin(), tampon.getOrientation());

        // update the omni visual node positions through the mappings
        if (m_omniVisualNode)
        {
            sofa::simulation::MechanicalPropagateOnlyPositionAndVelocityVisitor mechaVisitor(sofa::core::MechanicalParams::defaultInstance()); mechaVisitor.execute(m_omniVisualNode.get());
            sofa::simulation::UpdateMappingVisitor updateVisitor(sofa::core::ExecParams::defaultInstance()); updateVisitor.execute(m_omniVisualNode.get());
        }
    }
    d_posDevice.endEdit();
    d_angle.endEdit();
}


void GeomagicDriver::updateButtonStates(bool emitEvent)
{
    int nbrButton = 2;
    sofa::helper::fixed_array<bool, 2> buttons;
    buttons[0] = d_button_1.getValue();
    buttons[1] = d_button_2.getValue();

    //copy button state
    sofa::helper::fixed_array<bool, 2> oldStates;
    for (int i = 0; i < nbrButton; i++)
        oldStates[i] = buttons[i];

    buttons[0] = m_simuData.buttonState & HD_DEVICE_BUTTON_1;
    buttons[1] = m_simuData.buttonState & HD_DEVICE_BUTTON_2;

    d_button_1.setValue(buttons[0]);
    d_button_1.setValue(buttons[1]);

    // emit event if requested
    if (!emitEvent)
        return;

    sofa::simulation::Node::SPtr rootContext = static_cast<simulation::Node*>(this->getContext()->getRootContext());
    if (!rootContext)
    {
        msg_error() << "Rootcontext can't be found using this->getContext()->getRootContext()";
        return;
    }

    for (int i = 0; i < nbrButton; i++)
    {
        std::string eventString;
        if (buttons[i] && !oldStates[i]) // button pressed
            eventString = "button" + std::to_string(i) + "pressed";
        else if (!buttons[i] && oldStates[i]) // button released
            eventString = "button" + std::to_string(i) + "released";

        if (!eventString.empty())
        {
            sofa::core::objectmodel::ScriptEvent eventS(static_cast<simulation::Node*>(this->getContext()), eventString.c_str());
            rootContext->propagateEvent(core::ExecParams::defaultInstance(), &eventS);
        }
    }  
}

void GeomagicDriver::getMatrix(Mat<4,4, GLdouble> & M1, int index, double teta)
{
    const double ct = cos(teta);
    const double st = sin(teta);

    const Mat<4,4, GLdouble> & M = m_dh_matrices[index];

    M1[0][0] =  M[0][0]*ct + M[0][1]*st;
    M1[0][1] = -M[0][0]*st + M[0][1]*ct;
    M1[0][2] =  M[0][2];
    M1[0][3] =  M[0][3];

    M1[1][0] =  M[1][0]*ct + M[1][1]*st;
    M1[1][1] = -M[1][0]*st + M[1][1]*ct;
    M1[1][2] =  M[1][2];
    M1[1][3] =  M[1][3];

    M1[2][0] =  M[2][0]*ct + M[2][1]*st;
    M1[2][1] = -M[2][0]*st + M[2][1]*ct;
    M1[2][2] =  M[2][2];
    M1[2][3] =  M[2][3];

    M1[3][0] =  M[3][0]*ct + M[3][1]*st;
    M1[3][1] = -M[3][0]*st + M[3][1]*ct;
    M1[3][2] =  M[3][2];
    M1[3][3] =  M[3][3];
}

void GeomagicDriver::draw(const sofa::core::visual::VisualParams* vparams)
{
    if(m_errorDevice != 0)
        return;

    vparams->drawTool()->saveLastState();

    if (d_frameVisu.getValue() && m_initVisuDone)
    {
        vparams->drawTool()->disableLighting();

        float glRadius = (float)d_scale.getValue()*0.1f;
        vparams->drawTool()->drawArrow(m_posDeviceVisu[0].getCenter(), m_posDeviceVisu[0].getCenter() + m_posDeviceVisu[0].getOrientation().rotate(Vector3(2,0,0)*d_scale.getValue()), glRadius, Vec4f(1,0,0,1) );
        vparams->drawTool()->drawArrow(m_posDeviceVisu[0].getCenter(), m_posDeviceVisu[0].getCenter() + m_posDeviceVisu[0].getOrientation().rotate(Vector3(0,2,0)*d_scale.getValue()), glRadius, Vec4f(0,1,0,1) );
        vparams->drawTool()->drawArrow(m_posDeviceVisu[0].getCenter(), m_posDeviceVisu[0].getCenter() + m_posDeviceVisu[0].getOrientation().rotate(Vector3(0,0,2)*d_scale.getValue()), glRadius, Vec4f(0,0,1,1) );
    }

    if (d_omniVisu.getValue() && m_initVisuDone)
    {
        //Reactivate visual node
        if(!m_visuActive)
        {
            m_visuActive = true;

            for(int i=0; i<NVISUALNODE; i++)
            {
                m_omniVisualNode->addChild(visualNode[i].node);
                visualNode[i].node->updateContext();
            }
            m_omniVisualNode->updateContext();
        }

        VecCoord& posDOF =*(rigidDOF->x.beginEdit());
        posDOF.resize(m_posDeviceVisu.size());
        for(int i=0; i<NVISUALNODE+1; i++)
        {
            posDOF[i].getCenter() = m_posDeviceVisu[i].getCenter();
            posDOF[i].getOrientation() = m_posDeviceVisu[i].getOrientation();
        }

        //if buttons pressed, change stylus color
        std::string color = "grey";
        if(d_button_1.getValue())
        {
            if(d_button_2.getValue())
            {
                color = "yellow";
            }
            else
            {
                color = "blue";
            }
        }
        else if(d_button_2.getValue())
        {
            color = "red";
        }
        visualNode[0].visu->setColor(color);

        rigidDOF->x.endEdit();
    }
    else
    {
        if(m_visuActive)
        {
            m_visuActive = false;
            //delete omnivisual
            for(int i=0; i<NVISUALNODE; i++)
            {
                m_omniVisualNode->removeChild(visualNode[i].node);
            }
        }
    }

    vparams->drawTool()->restoreLastState();
}

void GeomagicDriver::computeBBox(const core::ExecParams*  params, bool  )
{
    if(m_errorDevice != 0)
        return;

    SReal minBBox[3] = {1e10,1e10,1e10};
    SReal maxBBox[3] = {-1e10,-1e10,-1e10};

    minBBox[0] = d_posDevice.getValue().getCenter()[0]-d_positionBase.getValue()[0]*d_scale.getValue();
    minBBox[1] = d_posDevice.getValue().getCenter()[1]-d_positionBase.getValue()[1]*d_scale.getValue();
    minBBox[2] = d_posDevice.getValue().getCenter()[2]-d_positionBase.getValue()[2]*d_scale.getValue();

    maxBBox[0] = d_posDevice.getValue().getCenter()[0]+d_positionBase.getValue()[0]*d_scale.getValue();
    maxBBox[1] = d_posDevice.getValue().getCenter()[1]+d_positionBase.getValue()[1]*d_scale.getValue();
    maxBBox[2] = d_posDevice.getValue().getCenter()[2]+d_positionBase.getValue()[2]*d_scale.getValue();

    this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<SReal>(minBBox,maxBBox));
}

void GeomagicDriver::handleEvent(core::objectmodel::Event *event)
{
    if(m_errorDevice != 0)
        return;

    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        if (m_hStateHandles.size() && m_hStateHandles[0] == HD_INVALID_HANDLE)
            return;

        m_simulationStarted = true;
        updatePosition();
    }
}

int GeomagicDriverClass = core::RegisterObject("Driver allowing interfacing with Geomagic haptic devices.")
.add< GeomagicDriver >()
.addAlias("DefaultHapticsDevice")
;

SOFA_DECL_CLASS(GeomagicDriver)

} // namespace controller

} // namespace component

} // namespace sofa
