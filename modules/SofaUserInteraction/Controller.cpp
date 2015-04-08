/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>


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
    if (dynamic_cast<sofa::simulation::AnimateBeginEvent *>(event))
    {
        onBeginAnimationStep((static_cast<sofa::simulation::AnimateBeginEvent *> (event))->getDt());
    }
    else if (dynamic_cast<sofa::simulation::AnimateEndEvent *>(event))
    {
        onEndAnimationStep((static_cast<sofa::simulation::AnimateEndEvent *> (event))->getDt());
    }
    else if (dynamic_cast<sofa::core::objectmodel::KeypressedEvent *>(event))
    {
        sofa::core::objectmodel::KeypressedEvent *kpev = dynamic_cast<sofa::core::objectmodel::KeypressedEvent *>(event);
        onKeyPressedEvent(kpev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::KeyreleasedEvent *>(event))
    {
        sofa::core::objectmodel::KeyreleasedEvent *krev = dynamic_cast<sofa::core::objectmodel::KeyreleasedEvent *>(event);
        onKeyReleasedEvent(krev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event))
    {
        sofa::core::objectmodel::MouseEvent *mev = dynamic_cast<sofa::core::objectmodel::MouseEvent *>(event);
        onMouseEvent(mev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::JoystickEvent *>(event))
    {
        sofa::core::objectmodel::JoystickEvent *jev = dynamic_cast<sofa::core::objectmodel::JoystickEvent *>(event);
        onJoystickEvent(jev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::HapticDeviceEvent *>(event))
    {
        sofa::core::objectmodel::HapticDeviceEvent *oev = dynamic_cast<sofa::core::objectmodel::HapticDeviceEvent *>(event);
        onHapticDeviceEvent(oev);
    }
    else if (dynamic_cast<sofa::core::objectmodel::GUIEvent *>(event))
    {
        sofa::core::objectmodel::GUIEvent *gev = dynamic_cast<sofa::core::objectmodel::GUIEvent *>(event);
        onGUIEvent(gev);
    }
}

} // namespace controller

} // namepace component

} // namespace sofa

