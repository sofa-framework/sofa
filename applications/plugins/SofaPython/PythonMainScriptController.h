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
#ifndef PYTHONMAINCONTROLLER_H
#define PYTHONMAINCONTROLLER_H

#include "PythonEnvironment.h"
#include "ScriptController.h"
#include <sofa/core/objectmodel/DataFileName.h>

/// Forward declarations
namespace sofa {
    namespace core{
        namespace objectmodel{
            class IdleEvent ;
        }
    }
}

namespace sofa
{

namespace component
{

namespace controller
{

/// very similar to PythonScriptController expect the callbacks are not in
/// a python object (class) but directly in the file
/// it is less powerful but less verbose
class SOFA_SOFAPYTHON_API PythonMainScriptController : public ScriptController
{
public:
    SOFA_CLASS(PythonMainScriptController,ScriptController);

protected:
    PythonMainScriptController( const char* filename );

    void handleEvent(core::objectmodel::Event *event) override;

    /// @name Script interface
    ///   Function that needs to be implemented for each script language
    /// Typically, all "script_*" functions call the corresponding "*" function of the script, if it exists
    /// @{

    virtual void loadScript() override;

    virtual void script_onLoaded(sofa::simulation::Node* node) override ;     /// called once, immediately after the script is loaded
    virtual void script_createGraph(sofa::simulation::Node* node) override ;  /// called when the script must create its graph
    virtual void script_initGraph(sofa::simulation::Node* node) override ;    /// called when the script must init its graph, once all the graph has been create
    virtual void script_bwdInitGraph(sofa::simulation::Node* node) override ; /// called when the script must init its graph, once all the graph has been create

    virtual void script_storeResetState() override;
    virtual void script_reset() override;

    virtual void script_cleanup() override ;

    /// keyboard & mouse events
    virtual bool script_onKeyPressed(const char c) override;
    virtual bool script_onKeyReleased(const char c) override ;

    virtual void script_onMouseButtonLeft(const int posX,const int posY,const bool pressed) override;
    virtual void script_onMouseButtonRight(const int posX,const int posY,const bool pressed) override;
    virtual void script_onMouseButtonMiddle(const int posX,const int posY,const bool pressed) override;
    virtual void script_onMouseWheel(const int posX,const int posY,const int delta) override;

    /// called each frame
    virtual void script_onBeginAnimationStep(const double dt) override ;
    virtual void script_onEndAnimationStep(const double dt) override;

    virtual void script_onGUIEvent(const char* controlID, const char* valueName, const char* value) override ;

    /// Script events; user data is implementation-dependant
    virtual void script_onScriptEvent(core::objectmodel::ScriptEvent* event) override ;

    /// drawing
    virtual void script_draw(const core::visual::VisualParams*) override ;

    virtual void script_onIdleEvent(const sofa::core::objectmodel::IdleEvent* event) override ;

    /// @}


public:
    const char* m_filename {nullptr} ;

    /// optionnal script entry points:
    PyObject *m_Func_onKeyPressed         {nullptr} ;
    PyObject *m_Func_onKeyReleased        {nullptr} ;
    PyObject *m_Func_onMouseButtonLeft    {nullptr} ;
    PyObject *m_Func_onMouseButtonRight   {nullptr} ;
    PyObject *m_Func_onMouseButtonMiddle  {nullptr} ;
    PyObject *m_Func_onMouseWheel         {nullptr} ;
    PyObject *m_Func_onGUIEvent           {nullptr} ;
    PyObject *m_Func_onScriptEvent        {nullptr} ;
    PyObject *m_Func_onBeginAnimationStep {nullptr} ;
    PyObject *m_Func_onEndAnimationStep   {nullptr} ;
    PyObject *m_Func_onLoaded             {nullptr} ;
    PyObject *m_Func_createGraph          {nullptr} ;
    PyObject *m_Func_initGraph            {nullptr} ;
    PyObject *m_Func_bwdInitGraph         {nullptr} ;
    PyObject *m_Func_storeResetState      {nullptr} ;
    PyObject *m_Func_reset                {nullptr} ;
    PyObject *m_Func_cleanup              {nullptr} ;
    PyObject *m_Func_draw                 {nullptr} ;
    PyObject *m_Func_onIdle               {nullptr} ;
private:
    PythonMainScriptController();
};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // PYTHONCONTROLLER_H
