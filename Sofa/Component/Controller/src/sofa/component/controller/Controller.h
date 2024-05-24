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
#pragma once
#include <sofa/component/controller/config.h>

#include <sofa/core/behavior/BaseController.h>

namespace sofa::core::objectmodel
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

namespace sofa::component::controller
{

/**
 * @brief Controller Class.
 * Interface of user interaction on SOFA Components.
 * Provides also an interface for BeginAnimation and EndAnimation events
 * launched at the beginning and the end of a time step.
 */
class SOFA_COMPONENT_CONTROLLER_API Controller : public core::behavior::BaseController
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
    ~Controller() override;
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
    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_CONTROLLER()
    Data < bool > handleEventTriggersUpdate;


    Data< bool > d_handleEventTriggersUpdate; ///< Event reception triggers object update

public:

    void handleEvent(core::objectmodel::Event *) override;
};

} //namespace sofa::component::collision
