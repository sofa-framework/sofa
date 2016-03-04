/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "PythonMacros.h"
#include "PythonEnvironment.h"
#include "ScriptController.h"
#include "ScriptEnvironment.h"

#include <sofa/core/objectmodel/GUIEvent.h>
#include <sofa/core/objectmodel/MouseEvent.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>

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

    //std::cout<<getName()<<" ScriptController::parse"<<std::endl;

    // load & bind script
    loadScript();
    // call script notifications...
    script_onLoaded( down_cast<simulation::Node>(getContext()) );
    script_createGraph( down_cast<simulation::Node>(getContext()) );

 //   ScriptEnvironment::initScriptNodes();
}

void ScriptController::init()
{
    Controller::init();
    // init the script
    script_initGraph( down_cast<simulation::Node>(getContext()) );
//    ScriptEnvironment::initScriptNodes();
}

void ScriptController::bwdInit()
{
    Controller::bwdInit();
    // init the script
    script_bwdInitGraph( down_cast<simulation::Node>(getContext()) );
//    ScriptEnvironment::initScriptNodes();
}

void ScriptController::storeResetState()
{
    Controller::storeResetState();
    // init the script
    script_storeResetState();
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::reset()
{
    Controller::reset();
    // init the script
    script_reset();
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::cleanup()
{
    Controller::cleanup();
    // init the script
    script_cleanup();
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::onBeginAnimationStep(const double dt)
{
    script_onBeginAnimationStep(dt);
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::onEndAnimationStep(const double dt)
{
    script_onEndAnimationStep(dt);
    ScriptEnvironment::initScriptNodes();
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
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::onKeyPressedEvent(core::objectmodel::KeypressedEvent * evt)
{
    if( script_onKeyPressed(evt->getKey()) ) evt->setHandled();
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::onKeyReleasedEvent(core::objectmodel::KeyreleasedEvent * evt)
{
    if( script_onKeyReleased(evt->getKey()) ) evt->setHandled();
    ScriptEnvironment::initScriptNodes();
}

void ScriptController::onGUIEvent(core::objectmodel::GUIEvent *event)
{
    script_onGUIEvent(event->getControlID().c_str(),
            event->getValueName().c_str(),
            event->getValue().c_str());
    ScriptEnvironment::initScriptNodes();
}


void ScriptController::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::ScriptEvent::checkEventType(event))
    {
        script_onScriptEvent(static_cast<core::objectmodel::ScriptEvent *> (event));
        ScriptEnvironment::initScriptNodes();
    }
    else Controller::handleEvent(event);
}

void ScriptController::draw(const core::visual::VisualParams* vis)
{
	script_draw(vis);
	ScriptEnvironment::initScriptNodes();
}

} // namespace controller

} // namespace component

} // namespace sofa



