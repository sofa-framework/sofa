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
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#include "ScriptController.h"
#include <sofa/core/objectmodel/GUIEvent.h>

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

    std::cout<<getName()<<" ScriptController::parse"<<std::endl;

    // load & bind script
    loadScript();
    // call script notifications...
    script_onLoaded( dynamic_cast<simulation::tree::GNode*>(getContext()) );
    script_createGraph( dynamic_cast<simulation::tree::GNode*>(getContext()) );
}

void ScriptController::init()
{
    Controller::init();
    // init the script
    script_initGraph( dynamic_cast<simulation::tree::GNode*>(getContext()) );
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

void ScriptController::onGUIEvent(core::objectmodel::GUIEvent *event)
{
    script_onGUIEvent(event->getControlID().c_str(),
            event->getValueName().c_str(),
            event->getValue().c_str());
}

} // namespace controller

} // namespace component

} // namespace sofa



