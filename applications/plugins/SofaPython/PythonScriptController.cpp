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
#include "PythonScriptController.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>

#include "Binding_Base.h"
#include "Binding_BaseContext.h"
#include "Binding_Node.h"
#include "Binding_PythonScriptController.h"
#include "ScriptEnvironment.h"
#include "PythonScriptEvent.h"

namespace sofa
{

namespace component
{

namespace controller
{

////////////////////// LIST OF FUNCTION TO OVERRIDE IN PYTHON //////////////////



std::vector<std::string> initMethodsToTest()
{
    std::vector<std::string> tmp(ScriptController::CONTROLLERFUNCTIONLIST_COUNT);

    tmp[ScriptController::DRAW] = "draw";
    tmp[ScriptController::ONBEGINANIMATIONSTEP] = "onBeginAnimationStep";
    tmp[ScriptController::ONENDANIMATIONSTEP] = "onEndAnimationStep";
    tmp[ScriptController::ONKEYPRESSED] = "onKeyPressed";
    tmp[ScriptController::ONKEYRELEASED] = "onKeyReleased";
    tmp[ScriptController::ONMOUSEMOVE] = "onMouseMove";
    tmp[ScriptController::ONMOUSEBUTTONLEFT] = "onMouseButtonLeft";
    tmp[ScriptController::ONMOUSEBUTTONRIGHT] = "onMouseButtonRight";
    tmp[ScriptController::ONMOUSEBUTTONMIDDLE] = "onMouseButtonMiddle";
    tmp[ScriptController::ONMOUSEWHEEL] = "onMouseWheel";
    tmp[ScriptController::ONGUIEVENT] = "onGUIEvent";
    tmp[ScriptController::ONSCRIPTEVENT] = "onScriptEvent";
    return tmp;
}
/// The mapping between the python name and the index in the structure.
static const std::vector<std::string> s_methodsToTest = initMethodsToTest();




int PythonScriptControllerClass = core::RegisterObject("A Sofa controller scripted in python")
        .add< PythonScriptController >()
        //.addAlias("PythonController")
        ;

SOFA_DECL_CLASS(PythonController)

PythonScriptController::PythonScriptController()
    : ScriptController()
    , m_filename(initData(&m_filename, "filename","Python script filename"))
    , m_classname(initData(&m_classname, "classname","Python class implemented in the script to instanciate for the controller"))
    , m_variables( initData( &m_variables, "variables", "Array of string variables (equivalent to a c-like argv)" ) )
    , m_ScriptControllerClass(0)
    , m_ScriptControllerInstance(0)
{
    // various initialization stuff here...
}

void PythonScriptController::loadScript()
{
    if(!sofa::simulation::PythonEnvironment::runFile(m_filename.getFullPath().c_str()))
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " object - "<<m_filename.getFullPath().c_str()<<" script load error." )
        return;
    }

    // classe
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));
    m_ScriptControllerClass = PyDict_GetItemString(pDict,m_classname.getValueString().c_str());
    if (!m_ScriptControllerClass)
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " load error (class \""<<m_classname.getValueString()<<"\" not found)." )
        return;
    }
    //std::cout << getName() << " class \""<<m_classname.getValueString()<<"\" found OK." << std::endl;


    // verify that the class is a subclass of PythonScriptController

    if (1!=PyObject_IsSubclass(m_ScriptControllerClass,(PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController)))
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " load error (class \""<<m_classname.getValueString()<<"\" does not inherit from \"Sofa.PythonScriptController\")." )
        return;
    }

    // créer l'instance de la classe

    m_ScriptControllerInstance = BuildPySPtr<Base>(this,(PyTypeObject*)m_ScriptControllerClass);

    if (!m_ScriptControllerInstance)
    {
        // LOAD ERROR
        SP_MESSAGE_ERROR( getName() << " load error (class \""<<m_classname.getValueString()<<"\" instanciation error)." )
        return;
    }
    //std::cout << getName() << " class \""<<m_classname.getValueString()<<"\" instanciation OK." << std::endl;


    // check which methods are implemented, not to call unimplemented functions
    for(unsigned int i=0;i<s_methodsToTest.size();i++){
        m_functionAvailables[i] = false;
        const std::string& fctname = s_methodsToTest[i] ;
        if(  PyObject_HasAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController),fctname.c_str() ) ){
            if( PyObject_RichCompareBool(
                        PyObject_GetAttrString(m_ScriptControllerClass, fctname.c_str()),
                        PyObject_GetAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController), fctname.c_str()),Py_NE) ){
                sout << "implementation found for: " << fctname << sendl ;
                m_functionAvailables[i] = true ;
            }
        }else{
            if( PyObject_HasAttrString(m_ScriptControllerInstance, fctname.c_str())) {
                sout << "implementation found for: " << fctname << sendl ;
                m_functionAvailables[i] = true ;
            }
        }
    }


/*
#define BIND_SCRIPT_FUNC(funcName) \
    { \
    m_Func_##funcName = PyDict_GetItemString(m_ScriptControllerInstanceDict,#funcName); \
            if (!PyCallable_Check(m_Func_##funcName)) \
                {m_Func_##funcName=0; std::cout<<#funcName<<" not found"<<std::endl;} \
            else \
                {std::cout<<#funcName<<" OK"<<std::endl;} \
    }


    // functions are also borrowed references
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
    BIND_SCRIPT_FUNC(onScriptEvent)
*/
}

//using namespace simulation::tree;
using namespace sofa::core::objectmodel;


// call a function that returns void
#define SP_CALL_OBJECTFUNC(func, ...) { \
    PyObject *res = PyObject_CallMethod(m_ScriptControllerInstance,func,__VA_ARGS__); \
    if (!res) \
    { \
        SP_MESSAGE_EXCEPTION( "in " << m_classname.getValueString() << "." << func ) \
        PyErr_Print(); \
    } \
    else \
        Py_DECREF(res); \
}


// call a function that returns a boolean
#define SP_CALL_OBJECTBOOLFUNC(func, ...) { \
    PyObject *res = PyObject_CallMethod(m_ScriptControllerInstance,func,__VA_ARGS__); \
    if (!res) \
    { \
        SP_MESSAGE_EXCEPTION( "in " << m_classname.getValueString() << "." << func ) \
        PyErr_Print(); \
    } \
    else \
    { \
        if PyBool_Check(res) b = ( res == Py_True ); \
        /*else SP_MESSAGE_WARNING("PythonScriptController::"<<func<<" should return a bool")*/ \
        Py_DECREF(res); \
    } \
}


void PythonScriptController::script_onLoaded(sofa::simulation::Node *node)
{
//    SP_CALL_MODULEFUNC(m_Func_onLoaded, "(O)", SP_BUILD_PYSPTR(node))
    SP_CALL_OBJECTFUNC(const_cast<char*>("onLoaded"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_createGraph(sofa::simulation::Node *node)
{
//    SP_CALL_MODULEFUNC(m_Func_createGraph, "(O)", SP_BUILD_PYSPTR(node))
    SP_CALL_OBJECTFUNC(const_cast<char*>("createGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_initGraph(sofa::simulation::Node *node)
{
//    SP_CALL_MODULEFUNC(m_Func_initGraph, "(O)", SP_BUILD_PYSPTR(node))
    SP_CALL_OBJECTFUNC(const_cast<char*>("initGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_bwdInitGraph(sofa::simulation::Node *node)
{
//    SP_CALL_MODULEFUNC(m_Func_initGraph, "(O)", SP_BUILD_PYSPTR(node))
    SP_CALL_OBJECTFUNC(const_cast<char*>("bwdInitGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

bool PythonScriptController::script_onKeyPressed(const char c)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    bool b = false;
    SP_CALL_OBJECTBOOLFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONKEYPRESSED].c_str()),const_cast<char*>("(c)"), c)
    return b;
}

bool PythonScriptController::script_onKeyReleased(const char c)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    bool b = false;
    SP_CALL_OBJECTBOOLFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONKEYRELEASED].c_str()),const_cast<char*>("(c)"), c)
    return b;
}

void PythonScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    PyObject *pyPressed = pressed? Py_True : Py_False;
//    SP_CALL_MODULEFUNC(m_Func_onMouseButtonLeft, "(iiO)", posX,posY,pyPressed)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONMOUSEBUTTONLEFT].c_str()),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    PyObject *pyPressed = pressed? Py_True : Py_False;
//    SP_CALL_MODULEFUNC(m_Func_onMouseButtonRight, "(iiO)", posX,posY,pyPressed)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONMOUSEBUTTONRIGHT].c_str()),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    PyObject *pyPressed = pressed? Py_True : Py_False;
//    SP_CALL_MODULEFUNC(m_Func_onMouseButtonMiddle, "(iiO)", posX,posY,pyPressed)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONMOUSEBUTTONMIDDLE].c_str()),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
//    SP_CALL_MODULEFUNC(m_Func_onMouseWheel, "(iii)", posX,posY,delta)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONMOUSEWHEEL].c_str()),const_cast<char*>("(iii)"), posX,posY,delta)
}


void PythonScriptController::script_onBeginAnimationStep(const double dt)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_AnimationStep_")+this->getName()).c_str() );
//    SP_CALL_MODULEFUNC(m_Func_onBeginAnimationStep, "(d)", dt)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONBEGINANIMATIONSTEP].c_str()),const_cast<char*>("(d)"), dt)
}

void PythonScriptController::script_onEndAnimationStep(const double dt)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_AnimationStep_")+this->getName()).c_str() );
//    SP_CALL_MODULEFUNC(m_Func_onEndAnimationStep, "(d)", dt)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONENDANIMATIONSTEP].c_str()),const_cast<char*>("(d)"), dt)
}

void PythonScriptController::script_storeResetState()
{
//    SP_CALL_MODULEFUNC_NOPARAM(m_Func_storeResetState)
    SP_CALL_OBJECTFUNC(const_cast<char*>("storeResetState"),0)
}

void PythonScriptController::script_reset()
{
//    SP_CALL_MODULEFUNC_NOPARAM(m_Func_reset)
    SP_CALL_OBJECTFUNC(const_cast<char*>("reset"),0)
}

void PythonScriptController::script_cleanup()
{
//    SP_CALL_MODULEFUNC_NOPARAM(m_Func_cleanup)
    SP_CALL_OBJECTFUNC(const_cast<char*>("cleanup"),0)
}

void PythonScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
//    SP_CALL_MODULEFUNC(m_Func_onGUIEvent,"(sss)",controlID,valueName,value)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONGUIEVENT].c_str()),const_cast<char*>("(sss)"),controlID,valueName,value)
}

void PythonScriptController::script_onScriptEvent(core::objectmodel::ScriptEvent* event)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );

    if(sofa::core::objectmodel::PythonScriptEvent::checkEventType(event))
    {
        core::objectmodel::PythonScriptEvent *pyEvent = static_cast<core::objectmodel::PythonScriptEvent*>(event);
//        SP_CALL_MODULEFUNC(m_Func_onScriptEvent,"(OsO)",SP_BUILD_PYSPTR(pyEvent->getSender().get()),pyEvent->getEventName().c_str(),pyEvent->getUserData())

        SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::ONSCRIPTEVENT].c_str()),const_cast<char*>("(OsO)"),SP_BUILD_PYSPTR(pyEvent->getSender().get()),const_cast<char*>(pyEvent->getEventName().c_str()),pyEvent->getUserData())
    }

}

void PythonScriptController::script_draw(const core::visual::VisualParams*)
{
//    SP_CALL_MODULEFUNC_NOPARAM(m_Func_storeResetState)
    SP_CALL_OBJECTFUNC(const_cast<char*>(s_methodsToTest[ScriptController::DRAW].c_str()),0)
}


void PythonScriptController::handleEvent(core::objectmodel::Event *event)
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

