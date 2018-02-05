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
#include "PythonScriptController.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/AdvancedTimer.h>
using sofa::helper::AdvancedTimer;


using sofa::core::objectmodel::Base;
using sofa::simulation::Node;

#include "Binding_PythonScriptController.h"
using sofa::simulation::PythonEnvironment;

#include "PythonScriptEvent.h"
using sofa::core::objectmodel::PythonScriptEvent;

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileMonitor ;
using sofa::helper::system::FileEventListener ;

#include <sofa/core/objectmodel/IdleEvent.h>
using sofa::core::objectmodel::IdleEvent ;

#include "PythonFactory.h"

//TODO(dmarchal): This have to be merged with the ScopedAdvancedTimer
struct ActivableScopedAdvancedTimer {
    const char* message;
    bool m_active ;
    Base* m_base;
    ActivableScopedAdvancedTimer(bool active, const char* message, Base* base)
        : message( message ), m_active(active), m_base(base)
    {
        if(m_active)
            AdvancedTimer::stepBegin(message, m_base);
    }

    ~ActivableScopedAdvancedTimer()
    {
        if(m_active)
            AdvancedTimer::stepEnd(message, m_base);
    }
};

namespace sofa
{

namespace component
{

namespace controller
{


class MyFileEventListener : public FileEventListener
{
    PythonScriptController* m_controller ;
public:
    MyFileEventListener(PythonScriptController* psc){
        m_controller = psc ;
    }

    virtual ~MyFileEventListener(){}

    virtual void fileHasChanged(const std::string& filepath){
        PythonEnvironment::gil lock {__func__} ;

        /// This function is called when the file has changed. Two cases have
        /// to be considered if the script was already loaded once or not.
        if(!m_controller->scriptControllerInstance()){
            m_controller->doLoadScript();
        }else{
            PythonEnvironment::gil state {__func__ } ;
            std::string file=filepath;
            SP_CALL_FILEFUNC(const_cast<char*>("onReimpAFile"),
                             const_cast<char*>("s"),
                             const_cast<char*>(file.data()));

            m_controller->refreshBinding();
        }
    }
};



int PythonScriptControllerClass = core::RegisterObject("A Sofa controller scripted in python")
        .add< PythonScriptController >()
        ;

SOFA_DECL_CLASS(PythonScriptController)

PythonScriptController::PythonScriptController()
    : ScriptController()
    , m_filename(initData(&m_filename, "filename",
                          "Python script filename"))
    , m_classname(initData(&m_classname, "classname",
                           "Python class implemented in the script to instanciate for the controller"))
    , m_variables(initData(&m_variables, "variables",
                           "Array of string variables (equivalent to a c-like argv)" ) )
    , m_timingEnabled(initData(&m_timingEnabled, true, "timingEnabled",
                               "Set this attribute to true or false to activate/deactivate the gathering"
                               " of timing statistics on the python execution time. Default value is set"
                               "to true." ))
    , m_doAutoReload( initData( &m_doAutoReload, false, "autoreload",
                                "Automatically reload the file when the source code is changed. "
                                "Default value is set to false" ) )
    , m_ScriptControllerClass(0)
    , m_ScriptControllerInstance(0)
{
    m_filelistener = new MyFileEventListener(this) ;
}

PythonScriptController::~PythonScriptController()
{
    if(m_filelistener)
    {
        FileMonitor::removeListener(m_filelistener) ;
        delete m_filelistener ;
    }
}


void PythonScriptController::setInstance(PyObject* instance) {
    PythonEnvironment::gil lock(__func__);
    
    // "trust me i'm an engineer"
    if( m_ScriptControllerInstance ) {
        Py_DECREF( m_ScriptControllerInstance );
    }

    m_ScriptControllerInstance = instance;

    // note: we don't use PyObject_Type as it returns a new reference which is
    // not handled correctly in loadScript
    m_ScriptControllerClass = (PyObject*)instance->ob_type;

    Py_INCREF( instance );

    refreshBinding();
}


void PythonScriptController::refreshBinding()
{
    BIND_OBJECT_METHOD(onLoaded)
            BIND_OBJECT_METHOD(createGraph)
            BIND_OBJECT_METHOD(initGraph)
            BIND_OBJECT_METHOD(bwdInitGraph)
            BIND_OBJECT_METHOD(onKeyPressed)
            BIND_OBJECT_METHOD(onKeyReleased)
            BIND_OBJECT_METHOD(onMouseButtonLeft)
            BIND_OBJECT_METHOD(onMouseButtonRight)
            BIND_OBJECT_METHOD(onMouseButtonMiddle)
            BIND_OBJECT_METHOD(onMouseWheel)
            BIND_OBJECT_METHOD(onBeginAnimationStep)
            BIND_OBJECT_METHOD(onEndAnimationStep)
            BIND_OBJECT_METHOD(storeResetState)
            BIND_OBJECT_METHOD(reset)
            BIND_OBJECT_METHOD(cleanup)
            BIND_OBJECT_METHOD(onGUIEvent)
            BIND_OBJECT_METHOD(onScriptEvent)
            BIND_OBJECT_METHOD(draw)
            BIND_OBJECT_METHOD(onIdle)
}

bool PythonScriptController::isDerivedFrom(const std::string& name, const std::string& module)
{
    PythonEnvironment::gil lock(__func__);    
    PyObject* moduleDict = PyModule_GetDict(PyImport_AddModule(module.c_str()));
    PyObject* controllerClass = PyDict_GetItemString(moduleDict, name.c_str());

    return 1 == PyObject_IsInstance(m_ScriptControllerInstance, controllerClass);
}

void PythonScriptController::loadScript()
{
    PythonEnvironment::gil lock(__func__);        
    if(m_doAutoReload.getValue())
    {
        FileMonitor::addFile(m_filename.getFullPath(), m_filelistener) ;
    }

    // if the filename is empty, the controller is supposed to be in an already loaded file
    // otherwise load the controller's file
    if( m_filename.isSet() && !m_filename.getRelativePath().empty() && !PythonEnvironment::runFile(m_filename.getFullPath().c_str()) )
    {
        msg_error() << " load error (file '"<<m_filename.getFullPath().c_str()<<"' not parsable)" ;
        return;
    }

    // classe
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));
    m_ScriptControllerClass = PyDict_GetItemString(pDict,m_classname.getValueString().c_str());
    if (!m_ScriptControllerClass)
    {
        msg_error() << " load error (class '"<<m_classname.getValueString()<<"' not found)." ;
        return;
    }

    // verify that the class is a subclass of PythonScriptController
    if (1!=PyObject_IsSubclass(m_ScriptControllerClass,(PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController)))
    {

        msg_error() << " load error (class '"<<m_classname.getValueString()<<"' does not inherit from 'Sofa.PythonScriptController')." ;
        return;
    }

    // créer l'instance de la classe
    m_ScriptControllerInstance = BuildPySPtr<Base>(this,(PyTypeObject*)m_ScriptControllerClass);

    if (!m_ScriptControllerInstance)
    {
        msg_error() << " load error (class '" <<m_classname.getValueString()<<"' instanciation error)." ;
        return;
    }

    refreshBinding();
}

void PythonScriptController::doLoadScript()
{
    loadScript() ;
}

void PythonScriptController::script_onIdleEvent(const IdleEvent* /*event*/)
{
    FileMonitor::updates(0);

    {
        PythonEnvironment::gil lock(__func__);
        SP_CALL_MODULEFUNC_NOPARAM(m_Func_onIdle) ;
    }

    /// Flush the console to avoid the sys.stdout.flush() in each script function.
    std::cout.flush() ;
    std::cerr.flush() ;
}

void PythonScriptController::script_onLoaded(Node *node)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_onLoaded, "(O)", sofa::PythonFactory::toPython(node))
}

void PythonScriptController::script_createGraph(Node *node)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_createGraph, "(O)", sofa::PythonFactory::toPython(node))
}

void PythonScriptController::script_initGraph(Node *node)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_initGraph, "(O)", sofa::PythonFactory::toPython(node))
}

void PythonScriptController::script_bwdInitGraph(Node *node)
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_bwdInitGraph, "(O)", sofa::PythonFactory::toPython(node))
}

bool PythonScriptController::script_onKeyPressed(const char c)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(), 
                                               "PythonScriptController_onKeyPressed", this);
    bool b = false;
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEBOOLFUNC(m_Func_onKeyPressed,"(c)", c);
    return b;
}

bool PythonScriptController::script_onKeyReleased(const char c)
{

    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(),
                                               "PythonScriptController_onKeyReleased", this);
    bool b = false;
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEBOOLFUNC(m_Func_onKeyReleased,"(c)", c);
    return b;
}

void PythonScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(), 
                                               "PythonScriptController_onMouseButtonLeft",this);
    PythonEnvironment::gil lock(__func__);    
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonLeft, "(iiO)", posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(), 
                                               "PythonScriptController_onMouseButtonRight", this);
    PythonEnvironment::gil lock(__func__);
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonRight, "(iiO)", posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(), 
                                               "PythonScriptController_onMouseButtonMiddle", this);
    PythonEnvironment::gil lock(__func__);
    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_MODULEFUNC(m_Func_onMouseButtonMiddle, "(iiO)", posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(),
                                               "PythonScriptController_onMouseWheel", this);
    PythonEnvironment::gil lock(__func__);
    SP_CALL_MODULEFUNC(m_Func_onMouseWheel, "(iii)", posX,posY,delta)
}


void PythonScriptController::script_onBeginAnimationStep(const double dt)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(),
                                               "PythonScriptController_onBeginAnimationStep", this);
    PythonEnvironment::gil lock(__func__);
    SP_CALL_MODULEFUNC(m_Func_onBeginAnimationStep, "(d)", dt)
}

void PythonScriptController::script_onEndAnimationStep(const double dt)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(),
                                               "PythonScriptController_onEndAnimationStep", this);
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC(m_Func_onEndAnimationStep, "(d)", dt)
}

void PythonScriptController::script_storeResetState()
{
    PythonEnvironment::gil lock(__func__);
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_storeResetState)
}

void PythonScriptController::script_reset()
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_reset)
}

void PythonScriptController::script_cleanup()
{
    PythonEnvironment::gil lock(__func__);    
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_cleanup)
}

void PythonScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(),
                                               "PythonScriptController_onGUIEvent", this);
    PythonEnvironment::gil lock(__func__);
    SP_CALL_MODULEFUNC(m_Func_onGUIEvent,"(sss)",controlID,valueName,value);
}

void PythonScriptController::script_onScriptEvent(core::objectmodel::ScriptEvent* event)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(), 
                                               "PythonScriptController_onScriptEvent", this);
    PythonEnvironment::gil lock(__func__);
    PythonScriptEvent *pyEvent = static_cast<PythonScriptEvent*>(event);
    SP_CALL_MODULEFUNC(m_Func_onScriptEvent,"(OsO)",
                       sofa::PythonFactory::toPython(pyEvent->getSender().get()),
                       pyEvent->getEventName().c_str(),pyEvent->getUserData());
}



void PythonScriptController::script_draw(const core::visual::VisualParams*)
{
    ActivableScopedAdvancedTimer advancedTimer(m_timingEnabled.getValue(), 
                                               "PythonScriptController_draw", this);
    PythonEnvironment::gil lock(__func__);
    SP_CALL_MODULEFUNC_NOPARAM(m_Func_draw);
}

void PythonScriptController::handleEvent(core::objectmodel::Event *event)
{
    if (PythonScriptEvent::checkEventType(event)) {
        script_onScriptEvent(static_cast<PythonScriptEvent *> (event));
    } else {
        ScriptController::handleEvent(event);
    }
}


} // namespace controller

} // namespace component

} // namespace sofa

