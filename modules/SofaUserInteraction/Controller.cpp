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
//
// C++ Implementation : Controller
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#include <SofaUserInteraction/Controller.h>
#include <sofa/core/visual/VisualParams.h>

#include <sofa/core/objectmodel/Event.h>
#include <sofa/core/objectmodel/JoystickEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/HapticDeviceEvent.h>
#include <sofa/core/objectmodel/GUIEvent.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>


namespace sofa
{

namespace component
{

namespace controller
{


Controller::Controller()
    : handleEventTriggersUpdate( initData(&handleEventTriggersUpdate, false, "handleEventTriggersUpdate", "Event handling frequency controls the controller update frequency" ) )
{

}

Controller::~Controller()
{

}

void Controller::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::simulation::AnimateBeginEvent::checkEventType(event))
    {
        onBeginAnimationStep((static_cast<sofa::simulation::AnimateBeginEvent *> (event))->getDt());
    }
    else if (sofa::simulation::AnimateEndEvent::checkEventType(event))
    {
        onEndAnimationStep((static_cast<sofa::simulation::AnimateEndEvent *> (event))->getDt());
    }
    else if (sofa::core::objectmodel::KeypressedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeypressedEvent *kpev = static_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        onKeyPressedEvent(kpev);
    }
    else if (sofa::core::objectmodel::KeyreleasedEvent::checkEventType(event))
    {
        sofa::core::objectmodel::KeyreleasedEvent *krev = static_cast<sofa::core::objectmodel::KeyreleasedEvent *>(event);
        onKeyReleasedEvent(krev);
    }
    else if (sofa::core::objectmodel::MouseEvent::checkEventType(event))
    {
        sofa::core::objectmodel::MouseEvent *mev = static_cast<sofa::core::objectmodel::MouseEvent *>(event);
        onMouseEvent(mev);
    }
    else if (sofa::core::objectmodel::JoystickEvent::checkEventType(event))
    {
        sofa::core::objectmodel::JoystickEvent *jev = static_cast<sofa::core::objectmodel::JoystickEvent *>(event);
        onJoystickEvent(jev);
    }
    else if (sofa::core::objectmodel::HapticDeviceEvent::checkEventType(event))
    {
        sofa::core::objectmodel::HapticDeviceEvent *oev = static_cast<sofa::core::objectmodel::HapticDeviceEvent *>(event);
        onHapticDeviceEvent(oev);
    }
    else if (sofa::core::objectmodel::GUIEvent::checkEventType(event))
    {
        sofa::core::objectmodel::GUIEvent *gev = static_cast<sofa::core::objectmodel::GUIEvent *>(event);
        onGUIEvent(gev);
    }
}

} // namespace controller

} // namepace component

} // namespace sofa

