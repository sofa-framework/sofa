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
#include "PythonScriptController.h"
#include "PythonMacros.h"
#include <sofa/core/ObjectFactory.h>

#include "Binding_Base.h"
#include "Binding_BaseContext.h"
#include "Binding_Node.h"

namespace sofa
{

namespace component
{

namespace controller
{


int PythonScriptControllerClass = core::RegisterObject("A Sofa controller scripted in python")
        .add< PythonScriptController >()
        //.addAlias("PythonController")
        ;

SOFA_DECL_CLASS(PythonController)



PythonScriptController::PythonScriptController()
    : ScriptController()
    , m_filename(initData(&m_filename, "filename","Python script filename"))
    , m_Script(0)
{
    // various initialization stuff here...
}




void PythonScriptController::loadScript()
{
    if (m_Script)
    {
        std::cout << getName() << " load ignored: script already loaded." << std::endl;
    }
    m_Script = sofa::simulation::PythonEnvironment::importScript(m_filename.getFullPath().c_str());
    if (!m_Script)
    {
        // LOAD ERROR
        std::cout << "<PYTHON> ERROR : "<<getName() << " object - "<<m_filename.getFullPath().c_str()<<" script load error." << std::endl;
        return;
    }

    // binder les différents points d'entrée du script

    // pDict is a borrowed reference; no need to release it
    m_ScriptDict = PyModule_GetDict(m_Script);

    // functions are also borrowed references
    if (!m_ScriptDict)
    {
        // LOAD ERROR
        std::cout << getName() << " load error (dictionnary not found)." << std::endl;
        return;
    }

#define BIND_SCRIPT_FUNC(funcName) \
    { \
        m_Func_##funcName = PyDict_GetItemString(m_ScriptDict,#funcName); \
        if (!PyCallable_Check(m_Func_##funcName)) m_Func_##funcName=0; \
    }

//    std::cout << "Binding functions of script \"" << m_filename.getFullPath().c_str() << "\"" << std::endl;
//    std::cout << "Number of dictionnay entries: "<< PyDict_Size(m_ScriptDict) << std::endl;
    BIND_SCRIPT_FUNC(onLoaded)
    BIND_SCRIPT_FUNC(createGraph)
    BIND_SCRIPT_FUNC(initGraph)
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

}

using namespace simulation::tree;
using namespace sofa::core::objectmodel;

void PythonScriptController::script_onLoaded(sofa::simulation::Node *node)
{
    SP_CALL(m_Func_onLoaded, "(O)", SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_createGraph(sofa::simulation::Node *node)
{
    SP_CALL(m_Func_createGraph, "(O)", SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_initGraph(sofa::simulation::Node *node)
{
    SP_CALL(m_Func_initGraph, "(O)", SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_onKeyPressed(const char c)
{
    SP_CALL(m_Func_onKeyPressed, "(c)", c)
}
void PythonScriptController::script_onKeyReleased(const char c)
{
    SP_CALL(m_Func_onKeyReleased, "(c)", c)
}

void PythonScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL(m_Func_onMouseButtonLeft, "(iiO)", posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL(m_Func_onMouseButtonRight, "(iiO)", posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL(m_Func_onMouseButtonMiddle, "(iiO)", posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    SP_CALL(m_Func_onMouseWheel, "(iii)", posX,posY,delta)
}


void PythonScriptController::script_onBeginAnimationStep(const double dt)
{
    SP_CALL(m_Func_onBeginAnimationStep, "(d)", dt)
}

void PythonScriptController::script_onEndAnimationStep(const double dt)
{
    SP_CALL(m_Func_onEndAnimationStep, "(d)", dt)
}

void PythonScriptController::script_storeResetState()
{
    SP_CALL_NOPARAM(m_Func_storeResetState)
}

void PythonScriptController::script_reset()
{
    SP_CALL_NOPARAM(m_Func_reset)
}

void PythonScriptController::script_cleanup()
{
    SP_CALL_NOPARAM(m_Func_cleanup)
}

void PythonScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    SP_CALL(m_Func_onGUIEvent,"(sss)",controlID,valueName,value)
}



} // namespace controller

} // namespace component

} // namespace sofa

