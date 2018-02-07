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
#include "PythonMacros.h"
#include "PythonEnvironment.h"
#include "ScriptController.h"

#include <sofa/core/objectmodel/GUIEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>


#include <sofa/core/objectmodel/IdleEvent.h>
using sofa::core::objectmodel::IdleEvent ;

using namespace sofa::simulation;
using namespace sofa::core::objectmodel;

namespace sofa
{

namespace component
{

namespace controller
{

ScriptController::ScriptController()
    : Controller()
{
    // various initialization stuff here...
    f_listening = true; // par défaut, on écoute les events sinon le script va pas servir à grand chose
}



void ScriptController::parse(sofa::core::objectmodel::BaseObjectDescription *arg)
{
    Controller::parse(arg);

    // load & bind script
    loadScript();
    // call script notifications...
    script_onLoaded( down_cast<simulation::Node>(getContext()) );
    script_createGraph( down_cast<simulation::Node>(getContext()) );
}

void ScriptController::init()
{
    Controller::init();
    // init the script
    script_initGraph( down_cast<simulation::Node>(getContext()) );
}

void ScriptController::bwdInit()
{
    Controller::bwdInit();
    // init the script
    script_bwdInitGraph( down_cast<simulation::Node>(getContext()) );
}

void ScriptController::storeResetState()
{
    Controller::storeResetState();
    // init the script
    script_storeResetState();
}

void ScriptController::reset()
{
    Controller::reset();
    // init the script
    script_reset();
}

void ScriptController::cleanup()
{
    Controller::cleanup();
    // init the script
    script_cleanup();
}

void ScriptController::onBeginAnimationStep(const double dt)
{
    script_onBeginAnimationStep(dt);
}

void ScriptController::onEndAnimationStep(const double dt)
{
    script_onEndAnimationStep(dt);
}

void ScriptController::onMouseEvent(core::objectmodel::MouseEvent * evt)
{
    switch(evt->getState())
    {
    case core::objectmodel::MouseEvent::Move:
        break;
    case core::objectmodel::MouseEvent::LeftPressed:
        script_onMouseButtonLeft(evt->getPosX(),evt->getPosY(),true);
        break;
    case core::objectmodel::MouseEvent::LeftReleased:
        script_onMouseButtonLeft(evt->getPosX(),evt->getPosY(),false);
        break;
    case core::objectmodel::MouseEvent::RightPressed:
        script_onMouseButtonRight(evt->getPosX(),evt->getPosY(),true);
        break;
    case core::objectmodel::MouseEvent::RightReleased:
        script_onMouseButtonRight(evt->getPosX(),evt->getPosY(),false);
        break;
    case core::objectmodel::MouseEvent::MiddlePressed:
        script_onMouseButtonMiddle(evt->getPosX(),evt->getPosY(),true);
        break;
    case core::objectmodel::MouseEvent::MiddleReleased:
        script_onMouseButtonMiddle(evt->getPosX(),evt->getPosY(),false);
        break;
    case core::objectmodel::MouseEvent::Wheel:
        script_onMouseWheel(evt->getPosX(),evt->getPosY(),evt->getWheelDelta());
        break;
    case core::objectmodel::MouseEvent::Reset:
        break;
    default:
        break;
    }
}

void ScriptController::onKeyPressedEvent(core::objectmodel::KeypressedEvent * evt)
{
    if( script_onKeyPressed(evt->getKey()) )
        evt->setHandled();
}

void ScriptController::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent * evt)
{
    if( script_onKeyReleased(evt->getKey()) )
        evt->setHandled();
}

void ScriptController::onGUIEvent(core::objectmodel::GUIEvent *event)
{
    script_onGUIEvent(event->getControlID().c_str(),
                      event->getValueName().c_str(),
                      event->getValue().c_str());
}


void ScriptController::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::ScriptEvent::checkEventType(event))
    {
        script_onScriptEvent(static_cast<core::objectmodel::ScriptEvent *> (event));
    }
    else if (sofa::core::objectmodel::IdleEvent::checkEventType(event))
    {
        script_onIdleEvent(static_cast<IdleEvent *> (event));
    }
    else
        Controller::handleEvent(event);
}

void ScriptController::draw(const core::visual::VisualParams* vis)
{
    script_draw(vis);
}

} // namespace controller

} // namespace component

} // namespace sofa



