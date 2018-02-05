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
#include "PythonMacros.h"
#include "PythonMainScriptController.h"
#include <sofa/core/ObjectFactory.h>
using sofa::core::RegisterObject;

#include <sofa/helper/AdvancedTimer.h>
using sofa::helper::ScopedAdvancedTimer;

using sofa::core::visual::VisualParams;

using sofa::simulation::PythonEnvironment;

#include "PythonScriptEvent.h"
using sofa::core::objectmodel::Event;
using sofa::core::objectmodel::ScriptEvent;
using sofa::core::objectmodel::PythonScriptEvent;

#include <sofa/helper/logging/Messaging.h>
#include "PythonFactory.h"

//TODO(dmarchal): Use the deactivable ScopedTimer


namespace sofa
{

namespace component
{

namespace controller
{

using sofa::core::objectmodel::IdleEvent ;

int PythonMainScriptControllerClass = RegisterObject("A Sofa controller scripted in python, looking for callbacks directly "
                                                     "in the file (not in a class like the more general and powerful "
                                                     "PythonScriptController")
        .add< PythonMainScriptController >();

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
    PythonEnvironment::gil lock(__func__);
    if(!PythonEnvironment::runFile(m_filename))
    {
        SP_MESSAGE_ERROR( getName() << " object - "<<m_filename<<" script load error." )
        return;
    }

    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));



    std::stringstream msg;
    msg << "Found callbacks: ";

    #define BIND_SCRIPT_FUNC_WITH_MESSAGE(funcName){\
        BIND_SCRIPT_FUNC(funcName)\
        if( m_Func_##funcName ) msg<<#funcName<<", ";\
        }

    BIND_SCRIPT_FUNC_WITH_MESSAGE(onLoaded)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(createGraph)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(initGraph)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(bwdInitGraph)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onKeyPressed)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onKeyReleased)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onMouseButtonLeft)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onMouseButtonRight)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onMouseButtonMiddle)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onMouseWheel)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onBeginAnimationStep)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onEndAnimationStep)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(storeResetState)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(reset)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(cleanup)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onGUIEvent)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onScriptEvent)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(draw)
    BIND_SCRIPT_FUNC_WITH_MESSAGE(onIdle)

    #undef BIND_SCRIPT_FUNC_WITH_MESSAGE

    msg_info() << msg.str();

}

void PythonMainScriptController::script_onIdleEvent(const IdleEvent* event)
{
    // there's no such thing as a macro being better than something ;-)    
    (void) event;
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_onIdle)
}

void PythonMainScriptController::script_onLoaded(sofa::simulation::Node *node)
{
    PythonEnvironment::gil lock(__func__);        
    SP_CALL_MODULEFUNC(m_Func_onLoaded,"(O)",sofa::PythonFactory::toPython(node))
}

void PythonMainScriptController::script_createGraph(sofa::simulation::Node *node)
{
    PythonEnvironment::gil lock(__func__);            
    SP_CALL_MODULEFUNC(m_Func_createGraph,"(O)",sofa::PythonFactory::toPython(node))
}

void PythonMainScriptController::script_initGraph(sofa::simulation::Node *node)
{
    PythonEnvironment::gil lock(__func__);            
    // no ScriptController::parse for a PythonMainScriptController
    // so call these functions here
    script_onLoaded( down_cast<simulation::Node>(getContext()) );
    script_createGraph( down_cast<simulation::Node>(getContext()) );

    SP_CALL_MODULEFUNC(m_Func_initGraph,"(O)",sofa::PythonFactory::toPython(node))
}

void PythonMainScriptController::script_bwdInitGraph(sofa::simulation::Node *node)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_bwdInitGraph,"(O)",sofa::PythonFactory::toPython(node))
}

bool PythonMainScriptController::script_onKeyPressed(const char c)
{
    PythonEnvironment::gil lock(__func__);    
    bool b = false;
    SP_CALL_MODULEBOOLFUNC(m_Func_onKeyPressed,"(c)", c);
    return b;
}
bool PythonMainScriptController::script_onKeyReleased(const char c)
{
    PythonEnvironment::gil lock(__func__);    
    bool b = false;
    SP_CALL_MODULEBOOLFUNC(m_Func_onKeyReleased,"(c)", c);
    return b;
}

void PythonMainScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    PythonEnvironment::gil lock(__func__);    
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonLeft,"(iiO)", posX,posY,pyPressed);
}

void PythonMainScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    PythonEnvironment::gil lock(__func__);    
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonRight,"(iiO)", posX,posY,pyPressed);
}

void PythonMainScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    PythonEnvironment::gil lock(__func__);    
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonMiddle,"(iiO)", posX,posY,pyPressed);
}

void PythonMainScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_onMouseWheel,"(iii)", posX,posY,delta);
}


void PythonMainScriptController::script_onBeginAnimationStep(const double dt)
{
    PythonEnvironment::gil lock(__func__);    
    helper::ScopedAdvancedTimer advancedTimer("PythonMainScriptController_AnimationStep");
    SP_CALL_MODULEFUNC(m_Func_onBeginAnimationStep,"(d)", dt)
}

void PythonMainScriptController::script_onEndAnimationStep(const double dt)
{
    PythonEnvironment::gil lock(__func__);    
    helper::ScopedAdvancedTimer advancedTimer("PythonMainScriptController_AnimationStep");
    SP_CALL_MODULEFUNC(m_Func_onEndAnimationStep,"(d)", dt)
}

void PythonMainScriptController::script_storeResetState()
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_storeResetState)
}

void PythonMainScriptController::script_reset()
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_reset)
}

void PythonMainScriptController::script_cleanup()
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_cleanup)
}

void PythonMainScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_onGUIEvent,"(sss)",controlID,valueName,value)
}

void PythonMainScriptController::script_onScriptEvent(ScriptEvent* event)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonMainScriptController_Event_")+this->getName()).c_str() );

    PythonScriptEvent *pyEvent = static_cast<PythonScriptEvent*>(event);
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_onScriptEvent,"(OsO)",sofa::PythonFactory::toPython(pyEvent->getSender().get()),const_cast<char*>(pyEvent->getEventName().c_str()),pyEvent->getUserData())
}

void PythonMainScriptController::script_draw(const VisualParams*)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_draw)
}

void PythonMainScriptController::handleEvent(Event *event)
{
    if (PythonScriptEvent::checkEventType(event))
    {
        script_onScriptEvent(static_cast<PythonScriptEvent *> (event));
    }
    else ScriptController::handleEvent(event);
}


} // namespace controller

} // namespace component

} // namespace sofa

