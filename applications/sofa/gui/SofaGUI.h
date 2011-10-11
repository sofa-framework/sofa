/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_SOFAGUI_H
#define SOFA_GUI_SOFAGUI_H

#include <sofa/simulation/common/Node.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/component/configurationsetting/ViewerSetting.h>
#include <sofa/component/configurationsetting/MouseButtonSetting.h>

#include <list>
//class QWidget;

#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_SOFAGUI
#	define SOFA_SOFAGUI_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#	define SOFA_SOFAGUI_API SOFA_IMPORT_DYNAMIC_LIBRARY
#endif


namespace sofa
{

namespace gui
{

class SOFA_SOFAGUI_API SofaGUI
{

public:

    /// @name methods each GUI must implement
    /// @{
    virtual int mainLoop()=0;
    virtual void redraw()=0;
    virtual int closeGUI()=0;
    virtual void setScene(sofa::simulation::Node* groot, const char* filename=NULL, bool temporaryFile=false)=0;
    virtual sofa::simulation::Node* currentSimulation()=0;
    /// @}

    virtual void configureGUI(sofa::simulation::Node* groot);

    /// @name methods to configure the GUI
    virtual void setViewerResolution(int /* width */, int /* height */) {};
    virtual void setFullScreen() {};
    virtual void setBackgroundColor(const defaulttype::Vector3& /*color*/) {};
    virtual void setBackgroundImage(const std::string& /*image*/) {};
    virtual void setDumpState(bool) {};
    virtual void setLogTime(bool) {};
    virtual void setExportState(bool) {};
#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void setTraceVisitors(bool) {};
#endif
    virtual void setRecordPath(const std::string & /*path*/) {};
    virtual void setGnuplotPath(const std::string & /*path*/) {};

    virtual void setViewerConfiguration(sofa::component::configurationsetting::ViewerSetting* /*viewerConf*/) {};
    virtual void setMouseButtonConfiguration(sofa::component::configurationsetting::MouseButtonSetting* /*button*/) {};
    /// @}

    void exportGnuplot(sofa::simulation::Node* node, std::string gnuplot_directory="");


    static std::string& GetGUIName() { return guiName; }
    static const char* GetProgramName() { return programName; }
    static void SetProgramName(const char* argv0) { if(argv0) programName = argv0;}

protected:
    SofaGUI();
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    virtual ~SofaGUI();
    static std::string guiName; // would like to make it const but not possible with the current implementation of RealGUI...
    static const char* programName;

};

} // namespace gui

} // namespace sofa

#endif
