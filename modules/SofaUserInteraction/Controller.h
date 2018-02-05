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
//
// C++ Interface: Controller
//
// Description:
//
//
// Author: Pierre-Jean Bensoussan, Digital Trainers (2008)
//
// Copyright: See COPYING file that comes with this distribution
//
//

#ifndef SOFA_COMPONENT_CONTROLLER_CONTROLLER_H
#define SOFA_COMPONENT_CONTROLLER_CONTROLLER_H
#include "config.h"

#include <sofa/core/behavior/BaseController.h>

namespace sofa
{
namespace core
{
namespace objectmodel
{

class Event;
class MouseEvent;
class HapticDeviceEvent;
class KeypressedEvent;
class KeyreleasedEvent;
class JoystickEvent;
class XitactEvent;
class GUIEvent;
}
}
}

namespace sofa
{

namespace component
{

namespace controller
{

/**
 * @brief Controller Class.
 * Interface of user interaction on SOFA Components.
 * Provides also an interface for BeginAnimation and EndAnimation events
 * launched at the beginning and the end of a time step.
 */
class SOFA_USER_INTERACTION_API Controller : public core::behavior::BaseController
{

public:
    SOFA_CLASS(Controller,core::behavior::BaseController);
protected:
    /**
    * @brief Default constructor.
    */
    Controller();

    /**
    * @brief Destructor.
    */
    virtual ~Controller();
public:
    /**
    * @brief Mouse event callback.
    */
    virtual void onMouseEvent(core::objectmodel::MouseEvent *) {}

    /**
    * @brief HapticDevice event callback.
    */
    virtual void onHapticDeviceEvent(core::objectmodel::HapticDeviceEvent *) {}

    /**
    * @brief Xitact event callback.
    */
    //virtual void onXitactEvent(core::objectmodel::HapticDeviceEvent *){}


    /**
    * @brief Key Press event callback.
    */
    virtual void onKeyPressedEvent(core::objectmodel::KeypressedEvent *) {}

    /**
    * @brief Key Release event callback.
    */
    virtual void onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent *) {}

    /**
    * @brief Joystick event callback.
    */
    virtual void onJoystickEvent(core::objectmodel::JoystickEvent *) {}

    /**
    * @brief Begin Animation event callback.
    */
    virtual void onBeginAnimationStep(const double /*dt*/) {}

    /**
    * @brief End Animation event callback.
    */
    virtual void onEndAnimationStep(const double /*dt*/) {}

    /**
    * @brief GUI event callback.
    */
    virtual void onGUIEvent(core::objectmodel::GUIEvent *) {}

protected:

    Data< bool > handleEventTriggersUpdate; ///< Event reception triggers object update ?

public:

    virtual void handleEvent(core::objectmodel::Event *) override;
};

} // namespace controller

} // namespace component

} // namespace sofa

#endif // SOFA_COMPONENT_CONTROLLER_CONTROLLER_H
