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

#include <Geomagic/GeomagicEmulator.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>

#include <sofa/core/objectmodel/ScriptEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>

#include <sofa/core/visual/VisualParams.h>
#include <Geomagic/GeomagicVisualModel.h>

#include <chrono>
#include <thread>

namespace sofa::component::controller
{
using namespace sofa::type;

GeomagicEmulatorTask::GeomagicEmulatorTask(GeomagicEmulator* ptr, CpuTask::Status* pStatus)
    :CpuTask(pStatus)
    , m_driver(ptr)
{

}

GeomagicEmulatorTask::MemoryAlloc GeomagicEmulatorTask::run()
{
    Vector3 currentForce;
    Vector3 pos_in_world;
    
    m_driver->lockPosition.lock();
    m_driver->m_simuData = m_driver->m_hapticData;
    m_driver->lockPosition.unlock();
    
    if (m_driver->m_terminate == false)
    {
        TaskScheduler::getInstance()->addTask(new GeomagicEmulatorTask(m_driver, &m_driver->_simStepStatus));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return MemoryAlloc::Dynamic;
}

//constructeur
GeomagicEmulator::GeomagicEmulator()
    : GeomagicDriver()
    , d_speedFactor(initData(&d_speedFactor, SReal(0.1), "speedFactor", "factor to increase/decrease the movements speed"))
    , _taskScheduler(nullptr)
    , m_terminate(false)
 {
    this->f_listening.setValue(true);
    m_forceFeedback = nullptr;
    m_GeomagicVisualModel = std::make_unique<GeomagicVisualModel>();

    oldStates[0] = false;
    oldStates[1] = false;
}


void GeomagicEmulator::clearDevice()
{
    m_terminate = true;
    while (_simStepStatus.isBusy())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    _taskScheduler->stop();
}



void GeomagicEmulator::initDevice()
{
    unsigned int mNbThread = 2;

    _taskScheduler = sofa::simulation::TaskScheduler::getInstance();
    _taskScheduler->init(mNbThread);
    _taskScheduler->addTask(new GeomagicEmulatorTask(this, &_simStepStatus));

    double init_jointAngles[3] = { 0.0, 0.26889, -0.370813 };
    double init_gimbalAngles[3] = { 0.000108409, 0.797273, -1.94046 };
    double init_transform[16] = { -0.361371, 0.848922, -0.385672, 0,
        -0.932422, -0.328976, 0.149546, 0,
        7.57409e-05, 0.413651, 0.910436, 0,
        0, -65.5107, -88.1142, 1 };

    for (int i = 0; i < 3; i++)
    {
        m_hapticData.angle1[i] = init_jointAngles[i];
        m_hapticData.angle2[i] = init_gimbalAngles[i];
    }

    for (int i = 0; i < 16; i++)
    {
        m_hapticData.transform[i] = init_transform[i];
    }

    // 2.6- Need to wait several ms for the scheduler to be well launched and retrieving correct device information before updating information on the SOFA side.
    std::this_thread::sleep_for(std::chrono::milliseconds(42));
    updatePosition();
}

void GeomagicEmulator::computeTransform()
{

}


void GeomagicEmulator::applyTranslation(sofa::type::Vec3 translation)
{
    lockPosition.lock();
    Vec3d & posDevice = *d_positionBase.beginEdit();
    const SReal& factor = d_speedFactor.getValue();
    posDevice += translation * factor;
    d_positionBase.endEdit();    
    lockPosition.unlock();
}



void GeomagicEmulator::worldToLocal(sofa::type::Vec3& vector)
{
    vector = d_orientationTool.getValue().rotate(vector);
}

void GeomagicEmulator::moveJoint1(SReal value)
{
    m_hapticData.angle1[0] += value;
    computeTransform();
}

void GeomagicEmulator::moveJoint2(SReal value)
{
    m_hapticData.angle1[1] += value;
    computeTransform();
}

void GeomagicEmulator::moveJoint3(SReal value)
{
    m_hapticData.angle1[2] += value;
    computeTransform();
}

void GeomagicEmulator::moveGimbalAngle1(SReal value)
{
    m_hapticData.angle2[0] += value;
    computeTransform();
}

void GeomagicEmulator::moveGimbalAngle2(SReal value)
{
    m_hapticData.angle2[1] += value;
    computeTransform();
}

void GeomagicEmulator::moveGimbalAngle3(SReal value)
{
    m_hapticData.angle2[2] += value;
    computeTransform();
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
        const SReal& speedfactor = d_speedFactor.getValue();

        if (ke->getKey() == '1')
            moveJoint1(speedfactor);
        else if (ke->getKey() == '3')
            moveJoint1(-speedfactor);
        else if (ke->getKey() == '4')
            moveJoint2(speedfactor);
        else if (ke->getKey() == '6')
            moveJoint2(-speedfactor);
        else if (ke->getKey() == '7')
            moveJoint3(speedfactor);
        else if (ke->getKey() == '9')
            moveJoint3(-speedfactor);

        else if (ke->getKey() == '-')
            moveGimbalAngle1(speedfactor);
        else if (ke->getKey() == '+')
            moveGimbalAngle1(-speedfactor);
        else if (ke->getKey() == '8')
            moveGimbalAngle2(speedfactor);
        else if (ke->getKey() == '2')
            moveGimbalAngle2(-speedfactor);
    }
}


int GeomagicEmulatorClass = core::RegisterObject("Driver allowing interfacing with Geomagic haptic devices.")
.add< GeomagicEmulator >()
;

} // namespace sofa::component::controller
