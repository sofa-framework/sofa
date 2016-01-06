/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef PYTHONMAINCONTROLLER_H
#define PYTHONMAINCONTROLLER_H

#include "PythonEnvironment.h"
#include "ScriptController.h"
#include <sofa/core/objectmodel/DataFileName.h>

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
    PythonMainScriptController();
    PythonMainScriptController( const char* filename );

    void handleEvent(core::objectmodel::Event *event);

    /// @name Script interface
    ///   Function that needs to be implemented for each script language
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

    /// @}


public:
    sofa::core::objectmodel::DataFileName d_filename;

};


} // namespace controller

} // namespace component

} // namespace sofa

#endif // PYTHONCONTROLLER_H
