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
#include "PythonMainScriptController.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>

#include "Binding_Base.h"
#include "Binding_BaseContext.h"
#include "Binding_Node.h"
#include "ScriptEnvironment.h"
#include "PythonScriptEvent.h"

#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace component
{

namespace controller
{

int PythonMainScriptControllerClass = core::RegisterObject("A Sofa controller scripted in python, looking for callbacks directly in the file (not in a class like the more general and powerful PythonScriptController")
        .add< PythonMainScriptController >()
        ;


PythonMainScriptController::PythonMainScriptController()
    : ScriptController()
    , m_filename(NULL)
{
    assert(false); // sould never be called
}

PythonMainScriptController::PythonMainScriptController(const char* filename)
    : ScriptController()
    , m_filename(filename)
{
    loadScript();
}




void PythonMainScriptController::loadScript()
{
    if(!sofa::simulation::PythonEnvironment::runFile(m_filename))
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " object - "<<m_filename<<" script load error." )
        return;
    }


    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));




    BIND_SCRIPT_FUNC(onLoaded)
    BIND_SCRIPT_FUNC(createGraph)
    BIND_SCRIPT_FUNC(initGraph)
    BIND_SCRIPT_FUNC(bwdInitGraph)
    BIND_SCRIPT_FUNC(onKeyPressed)
    BIND_SCRIPT_FUNC(onKeyReleased)
    BIND_SCRIPT_FUNC(onMouseButtonLeft)
    BIND_SCRIPT_FUNC(onMouseButtonRight)
    BIND_SCRIPT_FUNC(onMouseButtonMiddle)
    BIND_SCRIPT_FUNC(onMouseWheel)
    BIND_SCRIPT_FUNC(onBeginAnimationStep)
    BIND_SCRIPT_FUNC(onEndAnimationStep)
    BIND_SCRIPT_FUNC(storeResetState)
    BIND_SCRIPT_FUNC(reset)
    BIND_SCRIPT_FUNC(cleanup)
    BIND_SCRIPT_FUNC(onGUIEvent)
    BIND_SCRIPT_FUNC(onScriptEvent)
    BIND_SCRIPT_FUNC(draw)

}



void PythonMainScriptController::script_onLoaded(sofa::simulation::Node *node)
{
    SP_CALL_MODULEFUNC(m_Func_onLoaded,"(O)",SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_createGraph(sofa::simulation::Node *node)
{
    SP_CALL_MODULEFUNC(m_Func_createGraph,"(O)",SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_initGraph(sofa::simulation::Node *node)
{
    // no ScriptController::parse for a PythonMainScriptController
    // so call these functions here
    script_onLoaded( down_cast<simulation::Node>(getContext()) );
    script_createGraph( down_cast<simulation::Node>(getContext()) );

    SP_CALL_MODULEFUNC(m_Func_initGraph,"(O)",SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_bwdInitGraph(sofa::simulation::Node *node)
{
    SP_CALL_MODULEFUNC(m_Func_bwdInitGraph,"(O)",SP_BUILD_PYSPTR(node))
}

bool PythonMainScriptController::script_onKeyPressed(const char c)
{
    bool b = false;
    SP_CALL_MODULEBOOLFUNC(m_Func_onKeyPressed,"(c)", c)
    return b;
}
bool PythonMainScriptController::script_onKeyReleased(const char c)
{
    bool b = false;
    SP_CALL_MODULEBOOLFUNC(m_Func_onKeyReleased,"(c)", c)
    return b;
}

void PythonMainScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonLeft,"(iiO)", posX,posY,pyPressed)
}

void PythonMainScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonRight,"(iiO)", posX,posY,pyPressed)
}

void PythonMainScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonMiddle,"(iiO)", posX,posY,pyPressed)
}

void PythonMainScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    SP_CALL_MODULEFUNC(m_Func_onMouseWheel,"(iii)", posX,posY,delta)
}


void PythonMainScriptController::script_onBeginAnimationStep(const double dt)
{
    helper::ScopedAdvancedTimer advancedTimer("PythonMainScriptController_AnimationStep");
    SP_CALL_MODULEFUNC(m_Func_onBeginAnimationStep,"(d)", dt)
}

void PythonMainScriptController::script_onEndAnimationStep(const double dt)
{
    helper::ScopedAdvancedTimer advancedTimer("PythonMainScriptController_AnimationStep");
    SP_CALL_MODULEFUNC(m_Func_onEndAnimationStep,"(d)", dt)
}

void PythonMainScriptController::script_storeResetState()
{
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_storeResetState)
}

void PythonMainScriptController::script_reset()
{
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_reset)
}

void PythonMainScriptController::script_cleanup()
{
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_cleanup)
}

void PythonMainScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    SP_CALL_MODULEFUNC(m_Func_onGUIEvent,"(sss)",controlID,valueName,value)
}

void PythonMainScriptController::script_onScriptEvent(core::objectmodel::ScriptEvent* event)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonMainScriptController_Event_")+this->getName()).c_str() );

    core::objectmodel::PythonScriptEvent *pyEvent = static_cast<core::objectmodel::PythonScriptEvent*>(event);
    SP_CALL_MODULEFUNC(m_Func_onScriptEvent,"(OsO)",SP_BUILD_PYSPTR(pyEvent->getSender().get()),const_cast<char*>(pyEvent->getEventName().c_str()),pyEvent->getUserData())
}

void PythonMainScriptController::script_draw(const core::visual::VisualParams*)
{
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_draw)
}

void PythonMainScriptController::handleEvent(core::objectmodel::Event *event)
{
    if (sofa::core::objectmodel::PythonScriptEvent::checkEventType(event))
    {
        script_onScriptEvent(static_cast<core::objectmodel::PythonScriptEvent *> (event));
        simulation::ScriptEnvironment::initScriptNodes();
    }
    else ScriptController::handleEvent(event);
}


} // namespace controller

} // namespace component

} // namespace sofa

