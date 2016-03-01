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

#include <sofa/helper/system/FileMonitor.h>
using sofa::helper::system::FileMonitor ;
using sofa::helper::system::FileEventListener ;

#include <sofa/core/objectmodel/DrawEvent.h>
using sofa::core::objectmodel::DrawEvent ;

#include <sofa/core/objectmodel/MouseEvent.h>
using sofa::core::objectmodel::MouseEvent ;

#include <sofa/simulation/common/AnimateBeginEvent.h>
using sofa::simulation::AnimateBeginEvent ;

#include <sofa/simulation/common/AnimateEndEvent.h>
using sofa::simulation::AnimateEndEvent ;

#include <sofa/core/objectmodel/HeartBeatEvent.h>
using sofa::core::objectmodel::HeartBeatEvent ;

#include <sofa/core/objectmodel/KeypressedEvent.h>
using sofa::core::objectmodel::KeypressedEvent ;

#include <sofa/core/objectmodel/KeyreleasedEvent.h>
using sofa::core::objectmodel::KeyreleasedEvent ;

#include <sofa/simulation/common/Node.h>
using sofa::simulation::Node ;

#include "Binding_Base.h"
#include "Binding_BaseContext.h"
#include "Binding_Node.h"
#include "Binding_PythonScriptController.h"
#include "ScriptEnvironment.h"
#include "PythonScriptEvent.h"

#define EARLYQUIT
#define EARLYQUITF

namespace sofa
{

namespace component
{

namespace controller
{

// call a function that returns void
#define SP_CALL_FILEFUNC(func, ...){\
    PyObject* pDict = PyModule_GetDict(PyImport_AddModule("__main__"));\
    PyObject *pFunc = PyDict_GetItemString(pDict, func);\
    if (PyCallable_Check(pFunc))\
{\
    PyObject *res = PyObject_CallFunction(pFunc,__VA_ARGS__); \
    if( res )  Py_DECREF(res); \
}\
}

class MyFileEventListener : public FileEventListener
{
    PythonScriptController* m_controller ;
public:
    MyFileEventListener(PythonScriptController* psc){
        m_controller = psc ;
    }

    virtual ~MyFileEventListener(){}

    virtual void fileHasChanged(const std::string& filepath){
        /// This function is called when the file has changed. Two cases have
        /// to be considered if the script was already loaded once or not.
        if(!m_controller->scriptControllerInstance()){
           m_controller->doLoadScript();
        }else{
            std::string file=filepath;
            SP_CALL_FILEFUNC(const_cast<char*>("onReimpAFile"),
                             const_cast<char*>("s"),
                             const_cast<char*>(file.data()));
        }
    }
};

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
    , m_doAutoReload( initData( &m_doAutoReload, false, "autoreload", "Automatically reload the file when the source code is changed" ) )
    , m_ScriptControllerClass(0)
    , m_ScriptControllerInstance(0)
{
    // various initialization stuff here...
    m_filelistener = new MyFileEventListener(this) ;
}

PythonScriptController::~PythonScriptController()
{
    if(m_filelistener){
        FileMonitor::removeListener(m_filelistener) ;
        delete m_filelistener ;
    }
}

void PythonScriptController::loadScript()
{
    if(m_doAutoReload.getValue()){
        FileMonitor::addFile(m_filename.getFullPath(), m_filelistener) ;
    }

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

    std::vector<std::string> methodsToTest;
    methodsToTest.push_back("draw");
    methodsToTest.push_back("onBeginAnimationStep");
    methodsToTest.push_back("onEndAnimationStep");
    methodsToTest.push_back("onKeyPressed");
    methodsToTest.push_back("onKeyReleased");
    methodsToTest.push_back("onMouseMove");
    methodsToTest.push_back("onMouseButtonLeft");
    methodsToTest.push_back("onMouseButtonRight");
    methodsToTest.push_back("onMouseButtonMiddle");
    methodsToTest.push_back("onHeartBeatEvent");
    methodsToTest.push_back("onMouseWheel");

    std::map<std::string, size_t> methodToMask;
    methodToMask["draw"] = 1 << DrawEvent::getEventType() ;
    methodToMask["onBeginAnimationStep"] = 1 << AnimateBeginEvent::getEventType() ;
    methodToMask["onEndAnimationStep"] = 1 << AnimateEndEvent::getEventType() ;
    methodToMask["onMouseMove"] = 1 << MouseEvent::getEventType() ;
    methodToMask["onMouseButtonLeft"] = 1 << MouseEvent::getEventType() ;
    methodToMask["onMouseButtonRight"] = 1 << MouseEvent::getEventType() ;
    methodToMask["onMouseButtonMiddle"] = 1 << MouseEvent::getEventType() ;
    methodToMask["onMouseWheel"] = 1 << MouseEvent::getEventType() ;
    methodToMask["onHeartBeatEvent"] = 1 << HeartBeatEvent::getEventType() ;
    methodToMask["onKeyPressed"] = 1 << KeypressedEvent::getEventType() ;
    methodToMask["onKeyReleased"] = 1 << KeyreleasedEvent::getEventType() ;

    std::vector<std::string>::iterator it=methodsToTest.begin();
    std::vector<std::string>::iterator end=methodsToTest.end();

    m_mask = 0 ;
    for(;it!=end;++it){
        if(  PyObject_HasAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController),it->c_str() ) ){
            if( PyObject_RichCompareBool(
                    PyObject_GetAttrString(m_ScriptControllerClass, it->c_str()),
                    PyObject_GetAttrString((PyObject*)&SP_SOFAPYTYPEOBJECT(PythonScriptController), it->c_str()),Py_NE) ){
                std::cout << "HAS INSTANCE...." << *it << std::endl ;
                m_mask = m_mask | methodToMask[*it] ;
            }else{
                std::cout << "MISSING METHOD..." << *it << std::endl ;

            }
        }else{
            if( PyObject_HasAttrString(m_ScriptControllerInstance, it->c_str())) {
                std::cout << "HAS INSTANCE 2...." << *it << std::endl ;
            }else{
                std::cout << "MISSING METHOD 2..." << *it << std::endl ;
            }
        }
    }

    Node* node=(Node*)getContext() ;
    node->m_mask = m_mask ;
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

void PythonScriptController::doLoadScript()
{
    loadScript() ;
}

void PythonScriptController::script_onLoaded(sofa::simulation::Node *node)
{
    SP_CALL_OBJECTFUNC(const_cast<char*>("onLoaded"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_createGraph(sofa::simulation::Node *node)
{
    SP_CALL_OBJECTFUNC(const_cast<char*>("createGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_initGraph(sofa::simulation::Node *node)
{
    SP_CALL_OBJECTFUNC(const_cast<char*>("initGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

void PythonScriptController::script_bwdInitGraph(sofa::simulation::Node *node)
{
    SP_CALL_OBJECTFUNC(const_cast<char*>("bwdInitGraph"),const_cast<char*>("(O)"),SP_BUILD_PYSPTR(node))
}

bool PythonScriptController::script_onKeyPressed(const char c)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUITF;

    bool b = false;
    SP_CALL_OBJECTBOOLFUNC(const_cast<char*>("onKeyPressed"),const_cast<char*>("(c)"), c)
    return b;
}

bool PythonScriptController::script_onKeyReleased(const char c)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUITF;

    bool b = false;
    SP_CALL_OBJECTBOOLFUNC(const_cast<char*>("onKeyReleased"),const_cast<char*>("(c)"), c)
    return b;
}

void PythonScriptController::script_onMouseMove(const int posX,const int posY)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;

    SP_CALL_OBJECTFUNC(const_cast<char*>("onMouseMove"),const_cast<char*>("(ii)"), posX,posY)
}


void PythonScriptController::script_onMouseButtonLeft(const int posX,const int posY,const bool pressed)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;

    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_OBJECTFUNC(const_cast<char*>("onMouseButtonLeft"),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonRight(const int posX,const int posY,const bool pressed)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;

    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_OBJECTFUNC(const_cast<char*>("onMouseButtonRight"),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;

    PyObject *pyPressed = pressed? Py_True : Py_False;
    SP_CALL_OBJECTFUNC(const_cast<char*>("onMouseButtonMiddle"),const_cast<char*>("(iiO)"), posX,posY,pyPressed)
}

void PythonScriptController::script_onMouseWheel(const int posX,const int posY,const int delta)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;
    SP_CALL_OBJECTFUNC(const_cast<char*>("onMouseWheel"),const_cast<char*>("(iii)"), posX,posY,delta)
}


void PythonScriptController::script_onBeginAnimationStep(const double dt)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_AnimationStep_")+this->getName()).c_str() );
    EARLYQUIT;
    SP_CALL_OBJECTFUNC(const_cast<char*>("onBeginAnimationStep"),const_cast<char*>("(d)"), dt)
}

void PythonScriptController::script_onEndAnimationStep(const double dt)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_AnimationStep_")+this->getName()).c_str() );
    EARLYQUIT;
    SP_CALL_OBJECTFUNC(const_cast<char*>("onEndAnimationStep"),const_cast<char*>("(d)"), dt)
}

void PythonScriptController::script_storeResetState()
{
    EARLYQUIT;
    SP_CALL_OBJECTFUNC(const_cast<char*>("storeResetState"),0)
}

void PythonScriptController::script_reset()
{
    EARLYQUIT;
    SP_CALL_OBJECTFUNC(const_cast<char*>("reset"),0)
}

void PythonScriptController::script_cleanup()
{
    EARLYQUIT;

    SP_CALL_OBJECTFUNC(const_cast<char*>("cleanup"),0)
}

void PythonScriptController::script_onGUIEvent(const char* controlID, const char* valueName, const char* value)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;

    SP_CALL_OBJECTFUNC(const_cast<char*>("onGUIEvent"),const_cast<char*>("(sss)"),controlID,valueName,value)
}

void PythonScriptController::script_onScriptEvent(core::objectmodel::ScriptEvent* event)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_Event_")+this->getName()).c_str() );
    EARLYQUIT;

    if(sofa::core::objectmodel::PythonScriptEvent::checkEventType(event))
    {
        core::objectmodel::PythonScriptEvent *pyEvent = static_cast<core::objectmodel::PythonScriptEvent*>(event);
//        SP_CALL_MODULEFUNC(m_Func_onScriptEvent,"(OsO)",SP_BUILD_PYSPTR(pyEvent->getSender().get()),pyEvent->getEventName().c_str(),pyEvent->getUserData())

        SP_CALL_OBJECTFUNC(const_cast<char*>("onScriptEvent"),const_cast<char*>("(OsO)"),SP_BUILD_PYSPTR(pyEvent->getSender().get()),const_cast<char*>(pyEvent->getEventName().c_str()),pyEvent->getUserData())
    }

}

void PythonScriptController::script_onHeartBeatEvent(HeartBeatEvent* /*event*/)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_HeartBeat_")+this->getName()).c_str() );
    EARLYQUIT;

    FileMonitor::updates(0);
    SP_CALL_OBJECTFUNC(const_cast<char*>("onHeartBeatEvent"),0)

    // Flush the console to avoid the sys.stdout.flush() in each script function.
    std::cout.flush() ;
    std::cerr.flush() ;
}

void PythonScriptController::script_draw(const core::visual::VisualParams*)
{
    helper::ScopedAdvancedTimer advancedTimer( (std::string("PythonScriptController_draw_")+this->getName()).c_str() );
    EARLYQUIT;

    SP_CALL_OBJECTFUNC(const_cast<char*>("draw"),0)
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

