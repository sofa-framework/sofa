/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <Geomagic/GeomagicEmulator.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/core/objectmodel/ScriptEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace component
{

namespace controller
{

GeomagicEmulatorTask::GeomagicEmulatorTask(GeomagicEmulator* ptr, CpuTask::Status* pStatus)
    :CpuTask(pStatus)
    , m_driver(ptr)
{

}

GeomagicEmulatorTask::MemoryAlloc GeomagicEmulatorTask::run()
{
    Vector3 currentForce;
    Vector3 pos_in_world;
    bool contact = false;
    long long duration;
    
    if (m_driver->m_forceFeedback)
    {
        //Vector3 pos(driver->m_omniData.transform[12+0]*0.1,driver->m_omniData.transform[12+1]*0.1,driver->m_omniData.transform[12+2]*0.1);        
        m_driver->lockPosition.lock();
        pos_in_world = m_driver->d_positionBase.getValue();// +driver->d_orientationTool.getValue().rotate(pos*driver->d_scale.getValue());
        m_driver->lockPosition.unlock();

       // msg_info(m_driver) << "computeForce start: ";
        auto t1 = std::chrono::high_resolution_clock::now();
        m_driver->m_forceFeedback->computeForce(pos_in_world[0],pos_in_world[1],pos_in_world[2], 0, 0, 0, 0, currentForce[0], currentForce[1], currentForce[2]);
        auto t2 = std::chrono::high_resolution_clock::now();        
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        for (int i = 0; i < 3; i++)
        {
            if (currentForce[i] != 0.0)
            {
                contact = true;
                break;
            }
        }
    }
    
    m_driver->lockPosition.lock();
    if (contact)
    {        
        double maxInputForceFeedback = m_driver->d_maxInputForceFeedback.getValue();
        double norm = currentForce.norm();        
        
        msg_warning(m_driver) << "forceFeedback: " << currentForce << " | " << pos_in_world << " -> " << norm << " -> duration: " << duration;
        if (norm > maxInputForceFeedback) {
            msg_warning(m_driver) << "###################################################";
            msg_warning(m_driver) << "BAD forceFeedback: " << currentForce << " | " << pos_in_world << " -> " << norm << " -> duration: " << duration;
            currentForce = Vector3(0, 0, 0);
        }        
    }

    m_driver->m_toolForceFeedBack = currentForce * m_driver->d_forceScale.getValue();
    m_driver->m_isInContact = contact;    
    m_driver->m_toolPosition = m_driver->d_positionBase.getValue() + currentForce * m_driver->d_forceScale.getValue();
//    std::cout << "# m_toolPosition: " << m_driver->m_toolPosition << " | " << m_driver->d_positionBase.getValue() << std::endl;
    //msg_info(m_driver) << "computeForce end: ";
    m_driver->lockPosition.unlock();      

    if (m_driver->m_terminate == false)
    {
        TaskScheduler::getInstance()->addTask(new GeomagicEmulatorTask(m_driver, &m_driver->_simStepStatus));
        Sleep(100);
    }

    return MemoryAlloc::Dynamic;
}

//constructeur
GeomagicEmulator::GeomagicEmulator()
    : d_deviceName(initData(&d_deviceName, std::string("Default Device"), "deviceName","Name of device Configuration"))
    , d_positionBase(initData(&d_positionBase, Vec3d(0, 0, 0), "positionBase", "Position of the interface base in the scene world coordinates"))
    , d_orientationTool(initData(&d_orientationTool, Quat(0, 0, 0, 1), "orientationTool", "Orientation of the tool"))
    , d_scale(initData(&d_scale, 1.0, "scale","Default scale applied to the Phantom Coordinates"))
    , d_forceScale(initData(&d_forceScale, 1.0, "forceScale","Default forceScale applied to the force feedback. "))
    , d_posDevice(initData(&d_posDevice, "positionDevice", "position of the base of the part of the device"))
    , d_button_1(initData(&d_button_1,"button1","Button state 1"))
    , d_button_2(initData(&d_button_2,"button2","Button state 2"))    
    , d_toolNodeName(initData(&d_toolNodeName, "toolNodeName", "Node of the tool to activate deactivate"))
    , d_speedFactor(initData(&d_speedFactor, SReal(1.0), "speedFactor", "factor to increase/decrease the movements speed"))
    , d_maxInputForceFeedback(initData(&d_maxInputForceFeedback, double(10.0), "maxInputForceFeedback", "Maximum value of the normed input force feedback for device security"))
    , m_isInContact(false)
    , _taskScheduler(nullptr)
    , m_terminate(false)
 {
    this->f_listening.setValue(true);
    m_forceFeedback = NULL;

    oldStates[0] = false;
    oldStates[1] = false;
}

GeomagicEmulator::~GeomagicEmulator()
{
    clearDevice();
}

//executed once at the start of Sofa, initialization of all variables excepts haptics-related ones
void GeomagicEmulator::init()
{
    
}

void GeomagicEmulator::clearDevice()
{
    m_terminate = true;
    while (_simStepStatus.isBusy())
    {
        std::cout << "Waiting to finish" << std::endl;
        Sleep(1);
    }
    _taskScheduler->stop();
}



void GeomagicEmulator::bwdInit()
{
    simulation::Node *context = dynamic_cast<simulation::Node *>(this->getContext()); // access to current node
    m_forceFeedback = context->get<ForceFeedback>(this->getTags(), sofa::core::objectmodel::BaseContext::SearchRoot);
    
    initDevice();
}



void GeomagicEmulator::initDevice(int cptInitPass)
{
    //HDSchedulerHandle hStateHandle = HD_INVALID_HANDLE;
    //m_hHD = 1;
    std::cout << "GeomagicEmulator::initDevice" << std::endl;
    unsigned int mNbThread = 2;

    _taskScheduler = sofa::simulation::TaskScheduler::getInstance();
    _taskScheduler->init(mNbThread);
    _taskScheduler->addTask(new GeomagicEmulatorTask(this, &_simStepStatus));

    //hdMakeCurrentDevice(m_hHD);
    //hdEnable(HD_FORCE_OUTPUT);
    //hStateHandle = hdScheduleAsynchronous(stateEmulated, this, HD_MAX_SCHEDULER_PRIORITY);
    //m_hStateHandles.push_back(hStateHandle);
    
    //hdStartScheduler();

    updatePosition();
}

void GeomagicEmulator::updatePosition()
{
    Mat3x3d mrot;

    GeomagicEmulator::Coord & posDevice = *d_posDevice.beginEdit();    
    const Quat & orientationTool = d_orientationTool.getValue();
    const double & scale = d_scale.getValue();
    
    // for the moment
    //posDevice = d_positionBase.getValue();

    // update button state

    updateButtonStates(true);

    lockPosition.lock();

    posDevice[0] = m_toolPosition[0];
    posDevice[1] = m_toolPosition[1];
    posDevice[2] = m_toolPosition[2];
    //std::cout << "GeomagicEmulator::updatePosition m_toolPosition: " << m_toolPosition << " | " << d_positionBase.getValue() << std::endl;

//    d_positionBase.setValue(m_toolPosition);
   // Sleep(100);

    lockPosition.unlock();

    //Vector3 currentForce;
    //m_forceFeedback->computeForce(posDevice[0], posDevice[1], posDevice[2], 0, 0, 0, 0, currentForce[0], currentForce[1], currentForce[2]);
    /*
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

    */
    d_posDevice.endEdit();    
    
}


void GeomagicEmulator::updateButtonStates(bool emitEvent)
{
    int nbrButton = 2;
    sofa::helper::fixed_array<bool, 2> buttons;
    buttons[0] = d_button_1.getValue();
    buttons[1] = d_button_2.getValue();
   
        
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

        oldStates[i] = buttons[i];
    }
}



void GeomagicEmulator::applyTranslation(sofa::defaulttype::Vec3 translation)
{
    lockPosition.lock();
    Vec3d & posDevice = *d_positionBase.beginEdit();
    const SReal& factor = d_speedFactor.getValue();
    posDevice += translation * factor;
    d_positionBase.endEdit();    
    lockPosition.unlock();
}



void GeomagicEmulator::worldToLocal(sofa::defaulttype::Vec3& vector)
{
    vector = d_orientationTool.getValue().rotate(vector);
}


void GeomagicEmulator::moveUp()
{
    Vec3 vec(0, 1, 0);
    worldToLocal(vec);
    applyTranslation(vec);
}


void GeomagicEmulator::moveDown()
{
    Vec3 vec(0, -1, 0);
    worldToLocal(vec);
    applyTranslation(vec);
}


void GeomagicEmulator::moveLeft()
{
    Vec3 vec(-1, 0, 0);
    worldToLocal(vec);
    applyTranslation(vec);
}


void GeomagicEmulator::moveRight()
{
    Vec3 vec(1, 0, 0);
    worldToLocal(vec);
    applyTranslation(vec);
}

void GeomagicEmulator::moveForward()
{
    Vec3 vec(0, 0, -1);
    worldToLocal(vec);
    applyTranslation(vec);
}


void GeomagicEmulator::moveBackward()
{
    Vec3 vec(0, 0, 1);
    worldToLocal(vec);
    applyTranslation(vec);
}


void GeomagicEmulator::handleEvent(core::objectmodel::Event *event)
{
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {        
        updatePosition();
    }
    else if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent* ke = static_cast<sofa::core::objectmodel::KeypressedEvent*>(event);
        //msg_info() << "GeomagicEmulator handleEvent gets character '" << ke->getKey() << "'. ";

        if (ke->getKey() == '+')
            moveForward();
        else if (ke->getKey() == '-')
            moveBackward();
        else if (ke->getKey() == '8')
            moveUp();
        else if (ke->getKey() == '2')
            moveDown();
        else if (ke->getKey() == '4')
            moveLeft();
        else if (ke->getKey() == '6')
            moveRight();
        else if (ke->getKey() == '5')
            d_button_1.setValue(!d_button_1.getValue());
    }
}


void GeomagicEmulator::onKeyPressedEvent(core::objectmodel::KeypressedEvent *kEvent)
{
    //msg_info() << "GeomagicEmulator onKeyPressedEvent gets character '" << kEvent->getKey() << "'. ";

    if (kEvent->getKey() == '+')
        moveForward();
    else if (kEvent->getKey() == '-')
        moveBackward();
    else if (kEvent->getKey() == '8')
        moveUp();
    else if (kEvent->getKey() == '2')
        moveDown();
    else if (kEvent->getKey() == '4')
        moveLeft();
    else if (kEvent->getKey() == '6')
        moveRight();
}


void GeomagicEmulator::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *kEvent)
{

}



void GeomagicEmulator::draw(const sofa::core::visual::VisualParams* vparams)
{
    vparams->drawTool()->drawSphere(m_toolPosition, 0.1f, defaulttype::Vec4f(1.0, 1.0, 1.0, 1.0));
    vparams->drawTool()->drawLine(m_toolPosition, m_toolPosition + m_toolForceFeedBack, defaulttype::Vec4f(1.0, 0.0, 0.0f, 1.0));
}



int GeomagicEmulatorClass = core::RegisterObject("Driver allowing interfacing with Geomagic haptic devices.")
.add< GeomagicEmulator >()
;

} // namespace controller

} // namespace component

} // namespace sofa
