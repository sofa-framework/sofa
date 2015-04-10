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
#include "PythonMainScriptController.h"
#include "PythonMacros.h"
#include <sofa/core/ObjectFactory.h>

#include "Binding_Base.h"
#include "Binding_BaseContext.h"
#include "Binding_Node.h"
#include "ScriptEnvironment.h"

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
    , d_filename(initData(&d_filename, "filename","Python script filename"))
    , m_Script(0)
    , m_ScriptDict(0)
{}

PythonMainScriptController::PythonMainScriptController(const char* filename)
    : ScriptController()
    , d_filename(initData(&d_filename, "filename","Python script filename"))
    , m_Script(0)
    , m_ScriptDict(0)
{
    d_filename.setValue( filename );
}

void PythonMainScriptController::loadScript()
{
    if (m_Script) return;

    m_Script = sofa::simulation::PythonEnvironment::importScript(d_filename.getFullPath().c_str());

    if (!m_Script)
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " object - "<<d_filename.getFullPath().c_str()<<" script load error." )
        return;
    }

    // binder les différents points d'entrée du script

    // pDict is a borrowed reference; no need to release it
    m_ScriptDict = PyModule_GetDict(m_Script);
    if (!m_ScriptDict)
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " load error (dictionnary not found)." )
        return;
    }

}

using namespace sofa::core::objectmodel;


#define SP_CALL_FILEFUNC(func, ...){\
        loadScript(); \
        PyObject *pFunc = PyDict_GetItemString(m_ScriptDict, func);\
        if (PyCallable_Check(pFunc))\
        {\
            PyObject *res = PyObject_CallFunction(pFunc,__VA_ARGS__); \
            if( res )  Py_DECREF(res); \
        }\
    else {serr<<"SP_CALL_FILEFUNC "<<func<<" not callable"<<sendl;}\
}

void PythonMainScriptController::script_onLoaded(sofa::simulation::Node *node)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onLoaded"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_createGraph(sofa::simulation::Node *node)
{
    SP_CALL_FILEFUNC(const_cast<char*>("createGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_initGraph(sofa::simulation::Node *node)
{
    SP_CALL_FILEFUNC(const_cast<char*>("initGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_bwdInitGraph(sofa::simulation::Node *node)
{
    SP_CALL_FILEFUNC(const_cast<char*>("bwdInitGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonMainScriptController::script_onKeyPressed(const char c)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onKeyPressed"),const_cast<char*>("(c)"), c)
}
void PythonMainScriptController::script_onKeyReleased(const char c)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onKeyReleased"),const_cast<char*>("(c)"), c)
}

void PythonMainScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_FILEFUNC(const_cast<char*>("onMouseButtonLeft"),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonMainScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_FILEFUNC(const_cast<char*>("onMouseButtonRight"),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonMainScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_FILEFUNC(const_cast<char*>("onMouseButtonMiddle"),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonMainScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onMouseWheel"),const_cast<char*>("(iii)"), posX,posY,delta)
}


void PythonMainScriptController::script_onBeginAnimationStep(const double dt)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onBeginAnimationStep"),const_cast<char*>("(d)"), dt)
}

void PythonMainScriptController::script_onEndAnimationStep(const double dt)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onEndAnimationStep"),const_cast<char*>("(d)"), dt)
}

void PythonMainScriptController::script_storeResetState()
{
    SP_CALL_FILEFUNC(const_cast<char*>("storeResetState"),0)
}

void PythonMainScriptController::script_reset()
{
    SP_CALL_FILEFUNC(const_cast<char*>("reset"),0)
}

void PythonMainScriptController::script_cleanup()
{
    SP_CALL_FILEFUNC(const_cast<char*>("cleanup"),0)
}

void PythonMainScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    SP_CALL_FILEFUNC(const_cast<char*>("onGUIEvent"),const_cast<char*>("(sss)"),controlID,valueName,value)
}

void PythonMainScriptController::script_onScriptEvent(core::objectmodel::ScriptEvent* event)
{
    core::objectmodel::PythonScriptEvent *pyEvent = dynamic_cast<core::objectmodel::PythonScriptEvent*>(event);
    if (!pyEvent)
    {
        // ignore
    }
    else
    {
        SP_CALL_FILEFUNC(const_cast<char*>("onScriptEvent"),const_cast<char*>("(OsO)"),SP_BUILD_PYSPTR(pyEvent->getSender().get()),const_cast<char*>(pyEvent->getEventName().c_str()),pyEvent->getUserData())
    }

    //TODO
}

void PythonMainScriptController::script_draw(const core::visual::VisualParams*)
{
    SP_CALL_FILEFUNC(const_cast<char*>("draw"),0)
}



} // namespace controller

} // namespace component

} // namespace sofa

