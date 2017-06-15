/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef PYTHONCONTROLLER_H
#define PYTHONCONTROLLER_H

#include "PythonEnvironment.h"
#include "ScriptController.h"
#include <sofa/core/objectmodel/DataFileName.h>

/// Forward declarations
namespace sofa {
    namespace core{
        namespace objectmodel{ class IdleEvent ;}
    }
    namespace helper{
        namespace system{ class FileEventListener; }
    }
}


namespace sofa
{

namespace component
{

namespace controller
{

class SOFA_SOFAPYTHON_API PythonScriptController : public ScriptController
{
public:
    SOFA_CLASS(PythonScriptController,ScriptController);

    PyObject* scriptControllerInstance() const {return m_ScriptControllerInstance;}

    bool isDerivedFrom(const std::string& name, const std::string& module = "__main__");
    void doLoadScript();
    void refreshBinding();
protected:
    PythonScriptController();
    virtual ~PythonScriptController();

    void handleEvent(core::objectmodel::Event *event);

    /// @name Script interface
    ///   Function that need to be implemented for each script language
    /// Typically, all "script_*" functions call the corresponding "*" function of the script, if it exists
    /// @{

    virtual void loadScript();

    virtual void script_onLoaded(sofa::simulation::Node* node);   // called once, immediately after the script is loaded
    virtual void script_createGraph(sofa::simulation::Node* node);       // called when the script must create its graph
    virtual void script_initGraph(sofa::simulation::Node* node);         // called when the script must init its graph, once all the graph has been create
    virtual void script_bwdInitGraph(sofa::simulation::Node* node);         // called when the script must init its graph, once all the graph has been create

    virtual void script_storeResetState();
    virtual void script_reset();

    virtual void script_cleanup();

    /// keyboard & mouse events
    /// \returns true iff the event is handled (the event won't be sent to other components)
    virtual bool script_onKeyPressed(const char c);
    virtual bool script_onKeyReleased(const char c);

    virtual void script_onMouseButtonLeft(const int posX,const int posY,const bool pressed);
    virtual void script_onMouseButtonRight(const int posX,const int posY,const bool pressed);
    virtual void script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed);
    virtual void script_onMouseWheel(const int posX,const int posY,const int delta);

    /// called each frame
    virtual void script_onBeginAnimationStep(const double dt);
    virtual void script_onEndAnimationStep(const double dt);

    virtual void script_onGUIEvent(const char* controlID, const char* valueName, const char* value);

    /// Script events; user data is implementation-dependant
    virtual void script_onScriptEvent(core::objectmodel::ScriptEvent* event);

    /// drawing
    virtual void script_draw(const core::visual::VisualParams*);

    /// Idle event is sent a regular interval from the host application
    virtual void script_onIdleEvent(const sofa::core::objectmodel::IdleEvent* event);

    /// @}

public:
    sofa::core::objectmodel::DataFileName       m_filename;
    sofa::core::objectmodel::Data<std::string>  m_classname;
    sofa::core::objectmodel::Data< helper::vector< std::string > >  m_variables; // array of string variables (equivalent to a c-like argv), while waiting to have a better way to share variables
    sofa::core::objectmodel::Data<bool>         m_timingEnabled;
    sofa::core::objectmodel::Data<bool>         m_doAutoReload;

protected:
    sofa::helper::system::FileEventListener* m_filelistener ;

    PyObject *m_ScriptControllerClass;      // class implemented in the script to use to instanciate the python controller
    //PyObject *m_ScriptControllerInstanceDict;  // functions dictionnary
    PyObject *m_ScriptControllerInstance;   // instance of m_ScriptControllerClass

    // optionnal script entry points:
    PyObject *m_Func_onKeyPressed;
    PyObject *m_Func_onKeyReleased;
    PyObject *m_Func_onMouseButtonLeft;
    PyObject *m_Func_onMouseButtonRight;
    PyObject *m_Func_onMouseButtonMiddle;
    PyObject *m_Func_onMouseWheel;
    PyObject *m_Func_onGUIEvent;
    PyObject *m_Func_onScriptEvent;
    PyObject *m_Func_onBeginAnimationStep;
    PyObject *m_Func_onEndAnimationStep;
    PyObject *m_Func_onLoaded;
    PyObject *m_Func_createGraph;
    PyObject *m_Func_initGraph;
    PyObject *m_Func_bwdInitGraph;
    PyObject *m_Func_storeResetState;
    PyObject *m_Func_reset;
    PyObject *m_Func_cleanup;
    PyObject *m_Func_draw;
    PyObject *m_Func_onIdle;
};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // PYTHONCONTROLLER_H
