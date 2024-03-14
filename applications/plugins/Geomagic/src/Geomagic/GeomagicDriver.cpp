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

#include <Geomagic/GeomagicDriver.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/core/objectmodel/ScriptEvent.h>

#include <sofa/core/visual/VisualParams.h>
#include <Geomagic/GeomagicVisualModel.h>
#include <thread>
#include <chrono>

namespace sofa::component::controller
{
    
using namespace sofa::defaulttype;
using namespace sofa::type;

#if GEOMAGIC_HAVE_OPENHAPTICS
// Method to get the first error on the deck and if logError is not set to false will pop up full error message before returning the error code.
// Return HD_SUCCESS == 0 if no error.
HDerror catchHDError(bool logError = true)
{
    HDErrorInfo error;
    if (HD_DEVICE_ERROR(error = hdGetError()))
    {
        if (logError)
        {
            msg_error("GeomagicDriver::HDError") << "Device ID " << error.hHD << " returns error code " << error.errorCode
                << ": " << hdGetErrorString(error.errorCode);
        }
        return error.errorCode;
    }

    return HD_SUCCESS;
}


// Callback method to be executed by HD scheduler to copy the data from haptic own struct @sa m_hapticData to SOFA one @sa m_simuData.
HDCallbackCode HDCALLBACK copyDeviceDataCallback(void * pUserData)
{
    GeomagicDriver * driver = (GeomagicDriver *)pUserData;
    driver->m_simuData = driver->m_hapticData;
    return HD_CALLBACK_CONTINUE;
}


// Callback method to get the tool position and angles and compute the Force to apply to the tool
HDCallbackCode HDCALLBACK stateCallback(void * userData)
{
    HDErrorInfo error;
    GeomagicDriver * driver = (GeomagicDriver * ) userData;

    hdMakeCurrentDevice(driver->m_hHD);
    if (HD_DEVICE_ERROR(error = hdGetError())) return HD_CALLBACK_CONTINUE;

    hdBeginFrame(driver->m_hHD);
    if (HD_DEVICE_ERROR(error = hdGetError())) return HD_CALLBACK_CONTINUE;

    hdGetDoublev(HD_CURRENT_TRANSFORM, driver->m_hapticData.transform);

    //angles
    hdGetDoublev(HD_CURRENT_JOINT_ANGLES,driver->m_hapticData.angle1);
    hdGetDoublev(HD_CURRENT_GIMBAL_ANGLES,driver->m_hapticData.angle2);

    // Will only update the device position data and don't retrieve forceFeeedback
    if (!driver->m_simulationStarted) {
        hdEndFrame(driver->m_hHD);
        return HD_CALLBACK_CONTINUE;
    }

    // button status
    hdGetIntegerv(HD_CURRENT_BUTTONS, &driver->m_hapticData.buttonState);


    Vector3 currentForce;
    if (driver->m_forceFeedback)
    {
        Vector3 pos(driver->m_hapticData.transform[12+0]*0.1,driver->m_hapticData.transform[12+1]*0.1,driver->m_hapticData.transform[12+2]*0.1);
        Vector3 pos_in_world = driver->d_positionBase.getValue() + driver->d_orientationBase.getValue().rotate(pos*driver->d_scale.getValue());

        driver->m_forceFeedback->computeForce(pos_in_world[0],pos_in_world[1],pos_in_world[2], 0, 0, 0, 0, currentForce[0], currentForce[1], currentForce[2]);
        driver->m_isInContact = false;
        for (int i=0; i<3; i++)
            if (currentForce[i] != 0.0)
            {
                driver->m_isInContact = true;
                break;
            }
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

    GeomagicDriver::SHDdouble omni_force[3];
    omni_force[0] = force_in_omni[0];
    omni_force[1] = force_in_omni[1];
    omni_force[2] = force_in_omni[2];

    hdSetDoublev(HD_CURRENT_FORCE, omni_force);

    hdEndFrame(driver->m_hHD);

    return HD_CALLBACK_CONTINUE;
}

#endif

//constructeur
GeomagicDriver::GeomagicDriver()
    : d_deviceName(initData(&d_deviceName, std::string("Default Device"), "deviceName","Name of device Configuration"))
    , d_positionBase(initData(&d_positionBase, Vec3(0,0,0), "positionBase","Position of the device base in the SOFA scene world coordinates"))
    , d_orientationBase(initData(&d_orientationBase, Quat(0,0,0,1), "orientationBase","Orientation of the device base in the SOFA scene world coordinates"))
    , d_orientationTool(initData(&d_orientationTool, Quat(0,0,0,1), "orientationTool","Orientation of the tool in the SOFA scene world coordinates"))
    , d_scale(initData(&d_scale, 1.0, "scale", "Default scale applied to the Device coordinates"))
    , d_forceScale(initData(&d_forceScale, 1.0, "forceScale", "Default forceScale applied to the force feedback. "))
    , d_maxInputForceFeedback(initData(&d_maxInputForceFeedback, double(1.0), "maxInputForceFeedback", "Maximum value of the normed input force feedback for device security"))
    , d_inputForceFeedback(initData(&d_inputForceFeedback, Vec3(0, 0, 0), "inputForceFeedback", "Input force feedback in case of no LCPForceFeedback is found (manual setting)"))
    , d_manualStart(initData(&d_manualStart, false, "manualStart", "If true, will not automatically initDevice at component init phase."))
    , d_emitButtonEvent(initData(&d_emitButtonEvent, false, "emitButtonEvent", "If true, will send event through the graph when button are pushed/released"))
    , d_frameVisu(initData(&d_frameVisu, false, "drawDeviceFrame", "Visualize the frame corresponding to the device tooltip"))
    , d_omniVisu(initData(&d_omniVisu, false, "drawDevice", "Visualize the Geomagic device in the virtual scene"))    
    , d_posDevice(initData(&d_posDevice, "positionDevice", "position of the base of the part of the device"))
    , d_angle(initData(&d_angle, "angle", "Angluar values of joint (rad)"))
    , d_button_1(initData(&d_button_1,"button1","Button state 1"))
    , d_button_2(initData(&d_button_2,"button2","Button state 2"))    
    , l_forceFeedback(initLink("forceFeedBack", "link to the forceFeedBack component, if not set will search through graph and take first one encountered."))
    , m_simulationStarted(false)
    , m_isInContact(false)
    , m_hHD(HD_INVALID_HANDLE)
{
    this->f_listening.setValue(true);
    m_forceFeedback = nullptr;
    m_GeomagicVisualModel = std::make_unique<GeomagicVisualModel>();
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Loading);
}


GeomagicDriver::~GeomagicDriver()
{
    clearDevice();
}


//executed once at the start of Sofa, initialization of all variables excepts haptics-related ones
void GeomagicDriver::init()
{
    // 1- retrieve ForceFeedback component pointer
    if (l_forceFeedback.empty())
    {
        simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
        m_forceFeedback = context->get<sofa::component::haptics::ForceFeedback>(this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);
    }
    else
    {
        m_forceFeedback = l_forceFeedback.get();
    }

    if (!m_forceFeedback.get())
    {
        msg_warning() << "No forceFeedBack component found in the scene. Only the motion of the haptic tool will be simulated.";
    }


    // 2- init device and Hd scheduler
    if (d_manualStart.getValue() == false)
        initDevice();
}


void GeomagicDriver::clearDevice()
{
#if GEOMAGIC_HAVE_OPENHAPTICS
    if (m_hHD != HD_INVALID_HANDLE)
        hdMakeCurrentDevice(m_hHD);

    // stop scheduler first only if some works are registered
    if (!m_hStateHandles.empty() && s_schedulerRunning) {
        hdStopScheduler();
        s_schedulerRunning = false;
    }

    // unschedule valid tasks
    for (auto schedulerHandle : m_hStateHandles)
    {
        if (schedulerHandle != HD_INVALID_HANDLE) {
            hdUnschedule(schedulerHandle);
        }
    }
    m_hStateHandles.clear();

    // clear device if valid
    if (m_hHD != HD_INVALID_HANDLE)
    {
        hdDisableDevice(m_hHD);
        m_hHD = HD_INVALID_HANDLE;
    }
#endif
}


void GeomagicDriver::initDevice()
{
#if GEOMAGIC_HAVE_OPENHAPTICS
    HDSchedulerHandle hStateHandle = HD_INVALID_HANDLE;

    if (s_schedulerRunning) // need to stop scheduler if already running before any init
    {
        hdStopScheduler();
        s_schedulerRunning = false;
    }
    
    // 2.1- init device
    m_hHD = hdInitDevice(d_deviceName.getValue().c_str());

    // loop here in case of already used device
    if (catchHDError(false) == HD_DEVICE_ALREADY_INITIATED)
    {
        // double initialisation, this can occure if device has not been release due to bad program exit
        msg_warning() << "Device has already been initialized. SOFA Will clear current device using 'hdDisableDevice' and re-init the device properly using 'hdInitDevice'.";
        
        // Will Try clear and reinit device (10 times max): get device id in the current HD servo loop and try to release it and iterate on init
        HDerror tmpError = HD_DEVICE_ALREADY_INITIATED;
        int securityLoop = 0;
        while (tmpError == HD_DEVICE_ALREADY_INITIATED && securityLoop < 10)
        {
            // release device
            m_hHD = hdGetCurrentDevice();
            if (m_hHD != HD_INVALID_HANDLE)
            {
                hdDisableDevice(m_hHD);
                m_hHD = HD_INVALID_HANDLE;
            }

            // init device
            m_hHD = hdInitDevice(d_deviceName.getValue().c_str());
            tmpError = catchHDError(false);
        }

        if (tmpError != HD_SUCCESS) // failed
        {
            m_hHD = HD_INVALID_HANDLE;
            sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }
    
    // 2.2- check device calibration
    if (hdCheckCalibration() != HD_CALIBRATION_OK)
    {
        // Possible values are : HD_CALIBRATION_OK || HD_CALIBRATION_NEEDS_UPDATE || HD_CALIBRATION_NEEDS_MANUAL_INPUT
        msg_error() << "GeomagicDriver initialisation failed because device " << d_deviceName.getValue() << " is not calibrated. Calibration should be done using Geomagic Touch setup before using it in SOFA.";
        return;
    }
    
    // 2.3- Start scheduler
    hdEnable(HD_FORCE_OUTPUT);
    hdEnable(HD_MAX_FORCE_CLAMPING);
    hdEnable(HD_SOFTWARE_FORCE_LIMIT);
    
    hdStartScheduler();
    s_schedulerRunning = true;

    if (catchHDError())
    {
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }
    

    //hdMakeCurrentDevice(m_hHD);

    // 2.4- Add tasks in the scheduler using callback functions
    hStateHandle = hdScheduleAsynchronous(stateCallback, this, HD_MAX_SCHEDULER_PRIORITY);
    m_hStateHandles.push_back(hStateHandle);

    hStateHandle = hdScheduleAsynchronous(copyDeviceDataCallback, this, HD_MIN_SCHEDULER_PRIORITY);
    m_hStateHandles.push_back(hStateHandle);

    if (catchHDError())
    {
        sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
        return;
    }

    // 2.5- Set max forceFeedback security. 
    // TODO check if this can be replaced by: HD_MAX_FORCE_CLAMPING and HD_SOFTWARE_FORCE_LIMIT
    if (d_maxInputForceFeedback.isSet())
    {
        msg_info() << "maxInputForceFeedback value (" << d_maxInputForceFeedback.getValue() << ") is set, carefully set the max force regarding your haptic device";

        if (d_maxInputForceFeedback.getValue() <= 0.0)
        {
            msg_error() << "maxInputForceFeedback value (" << d_maxInputForceFeedback.getValue() << ") is negative or 0, it should be strictly positive";
            d_maxInputForceFeedback.setValue(0.0);
        }
    }
    
    // 2.6- Need to wait several ms for the scheduler to be well launched and retrieving correct device information before updating information on the SOFA side.
    std::this_thread::sleep_for(std::chrono::milliseconds(42));
    updatePosition();
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
#else
    sofa::core::objectmodel::BaseObject::d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
#endif
}


void GeomagicDriver::updatePosition()
{
    type::Vec6 & angle = *d_angle.beginEdit();
    GeomagicDriver::Coord & posDevice = *d_posDevice.beginEdit();

    const Vec3 & positionBase = d_positionBase.getValue();
    const Quat & orientationBase = d_orientationBase.getValue();
    const Quat & orientationTool = d_orientationTool.getValue();
    const double & scale = d_scale.getValue();

    // update button state
    updateButtonStates();

    //copy angle
    angle[0] = m_simuData.angle1[0];
    angle[1] = m_simuData.angle1[1];
    angle[2] = -(M_PI/2)+m_simuData.angle1[2]-m_simuData.angle1[1];
    angle[3] = -(M_PI/2)-m_simuData.angle2[0];
    angle[4] = m_simuData.angle2[1];
    angle[5] = -(M_PI/2)-m_simuData.angle2[2];

    //copy the position of the tool
    Vec3 position;
    position[0] = m_simuData.transform[12+0] * 0.1;
    position[1] = m_simuData.transform[12+1] * 0.1;
    position[2] = m_simuData.transform[12+2] * 0.1;

    //copy rotation of the tool
    type::Mat3x3d mrot;
    for (int u=0; u<3; u++)
        for (int j=0; j<3; j++)
            mrot[u][j] = m_simuData.transform[j*4+u];

    Quat orientation;
    orientation.fromMatrix(mrot);

    //compute the position of the tool (according to positionbase, orientation base and the scale
    posDevice.getCenter() = positionBase + orientationBase.rotate(position*scale);
    posDevice.getOrientation() = orientationBase * orientation * orientationTool;

    d_posDevice.endEdit();
    d_angle.endEdit();


    if (d_omniVisu.getValue() && m_GeomagicVisualModel != nullptr)
    {
        if (!m_GeomagicVisualModel->isDisplayInitiate()) // first time, need to init visualModel first
        {
            sofa::simulation::Node::SPtr rootContext = static_cast<simulation::Node*>(this->getContext()->getRootContext());
            m_GeomagicVisualModel->initDisplay(rootContext, d_deviceName.getValue(), d_scale.getValue());            
        }

        if (!m_GeomagicVisualModel->isDisplayActivated())
            m_GeomagicVisualModel->activateDisplay(true);

        m_GeomagicVisualModel->updateDisplay(d_posDevice.getValue(), m_hapticData.angle1, m_hapticData.angle2);
    }
    else if (d_omniVisu.getValue() == false && m_GeomagicVisualModel && m_GeomagicVisualModel->isDisplayActivated())
    {
        m_GeomagicVisualModel->activateDisplay(false);
    }
}


void GeomagicDriver::updateButtonStates()
{
    int nbrButton = 2;
    sofa::type::fixed_array<bool, 2> buttons;
    buttons[0] = d_button_1.getValue();
    buttons[1] = d_button_2.getValue();

    //copy button state
    sofa::type::fixed_array<bool, 2> oldStates;
    for (int i = 0; i < nbrButton; i++)
        oldStates[i] = buttons[i];

    // get new values
#if GEOMAGIC_HAVE_OPENHAPTICS
    buttons[0] = m_simuData.buttonState & HD_DEVICE_BUTTON_1;
    buttons[1] = m_simuData.buttonState & HD_DEVICE_BUTTON_2;
#endif

    d_button_1.setValue(buttons[0]);
    d_button_2.setValue(buttons[1]);

    // emit event if requested
    if (!d_emitButtonEvent.getValue())
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


void GeomagicDriver::draw(const sofa::core::visual::VisualParams* vparams)
{
    if (sofa::core::objectmodel::BaseObject::d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
        return;

    vparams->drawTool()->saveLastState();

    if (d_frameVisu.getValue())
    {
        vparams->drawTool()->disableLighting();

        const GeomagicDriver::Coord& posDevice = d_posDevice.getValue();

        float glRadius = (float)d_scale.getValue()*0.1f;
        vparams->drawTool()->drawArrow(posDevice.getCenter(), posDevice.getCenter() + posDevice.getOrientation().rotate(type::Vector3(2,0,0)*d_scale.getValue()), glRadius, sofa::type::RGBAColor::red() );
        vparams->drawTool()->drawArrow(posDevice.getCenter(), posDevice.getCenter() + posDevice.getOrientation().rotate(type::Vector3(0,2,0)*d_scale.getValue()), glRadius, sofa::type::RGBAColor::green() );
        vparams->drawTool()->drawArrow(posDevice.getCenter(), posDevice.getCenter() + posDevice.getOrientation().rotate(type::Vector3(0,0,2)*d_scale.getValue()), glRadius, sofa::type::RGBAColor::blue() );
    }

    if (d_omniVisu.getValue() && m_GeomagicVisualModel != nullptr)
        m_GeomagicVisualModel->drawDevice(d_button_1.getValue(), d_button_2.getValue());

    vparams->drawTool()->restoreLastState();
}


void GeomagicDriver::computeBBox(const core::ExecParams*  params, bool  )
{
    SReal minBBox[3] = {1e10,1e10,1e10};
    SReal maxBBox[3] = {-1e10,-1e10,-1e10};

    minBBox[0] = d_posDevice.getValue().getCenter()[0]-d_positionBase.getValue()[0]*d_scale.getValue();
    minBBox[1] = d_posDevice.getValue().getCenter()[1]-d_positionBase.getValue()[1]*d_scale.getValue();
    minBBox[2] = d_posDevice.getValue().getCenter()[2]-d_positionBase.getValue()[2]*d_scale.getValue();

    maxBBox[0] = d_posDevice.getValue().getCenter()[0]+d_positionBase.getValue()[0]*d_scale.getValue();
    maxBBox[1] = d_posDevice.getValue().getCenter()[1]+d_positionBase.getValue()[1]*d_scale.getValue();
    maxBBox[2] = d_posDevice.getValue().getCenter()[2]+d_positionBase.getValue()[2]*d_scale.getValue();

    this->f_bbox.setValue(sofa::type::TBoundingBox<SReal>(minBBox,maxBBox));
}


void GeomagicDriver::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        if (sofa::core::objectmodel::BaseObject::d_componentState.getValue() != sofa::core::objectmodel::ComponentState::Valid)
            return;

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

} // namespace sofa::component::controller
