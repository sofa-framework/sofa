/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GUI_SOFAGUI_H
#define SOFA_GUI_SOFAGUI_H

#include "BaseViewer.h"
#include <sofa/simulation/common/Node.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/component/configurationsetting/ViewerSetting.h>
#include <sofa/component/configurationsetting/MouseButtonSetting.h>

#include <sofa/helper/system/config.h>
#include <list>

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
    /// Start the GUI loop
    virtual int mainLoop()=0;
    /// Update the GUI
    virtual void redraw()=0;
    /// Close the GUI
    virtual int closeGUI()=0;
    /// Register the scene in our GUI
    virtual void setScene(sofa::simulation::Node::SPtr groot, const char* filename=NULL, bool temporaryFile=false)=0;
    /// Get the rootNode of the sofa scene
    virtual sofa::simulation::Node* currentSimulation() = 0;
    /// @}

    /// Use a component setting to configure our GUI
    virtual void configureGUI(sofa::simulation::Node::SPtr groot);

    /// @name methods to configure the GUI
    /// @{
    virtual void setDumpState(bool) {};
    virtual void setLogTime(bool) {};
    virtual void setExportState(bool) {};
#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void setTraceVisitors(bool) {};
#endif
    virtual void setRecordPath(const std::string & /*path*/) {};
    virtual void setGnuplotPath(const std::string & /*path*/) {};

    virtual void registerViewer(sofa::gui::BaseViewer* /*_viewer*/) {}
    virtual void initViewer() {}
    virtual void setViewerConfiguration(sofa::component::configurationsetting::ViewerSetting* /*viewerConf*/) {};
    virtual void setViewerResolution(int /* width */, int /* height */) {};
    virtual void setFullScreen() {};
    virtual void setBackgroundColor(const defaulttype::Vector3& /*color*/) {};
    virtual void setBackgroundImage(const std::string& /*image*/) {};
    virtual void registerViewer(BaseViewer* viewer) {};

    virtual void setMouseButtonConfiguration(sofa::component::configurationsetting::MouseButtonSetting* /*button*/) {};
    /// @}

    /// @name methods to communicate with the GUI
    /// @{
    virtual void sendMessage(const std::string & /*msgType*/,const std::string & /*msgValue*/) {}
    /// @}

    void exportGnuplot(sofa::simulation::Node* node, std::string gnuplot_directory="");

    static std::string& GetGUIName() { return mGuiName; }

    static const char* GetProgramName() { return mProgramName; }
    static void SetProgramName(const char* argv0) { if(argv0) mProgramName = argv0;}

protected:
    SofaGUI();
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    virtual ~SofaGUI();

    static std::string mGuiName; // would like to make it const but not possible with the current implementation of RealGUI...
    static const char* mProgramName;
};


class BaseViewer;


class SofaGUIImpl : public SofaGUI
{
public:
    SofaGUIImpl() : mCreateViewerOpt(false)
    {}
    virtual ~SofaGUIImpl()
    {}

    virtual sofa::simulation::Node* currentSimulation() {if(mViewer)return mViewer->getScene();}
    virtual void registerViewer(sofa::gui::BaseViewer* viewer) {mViewer = viewer;}
    virtual void removeViewer()
    {
        if(mCreateViewersOpt)
        {
            delete mViewer;
            mViewer = NULL;
        }
    }
//    virtual void createViewers(const char* viewerName, viewer::SofaViewerArgument arg);
//    virtual void initViewer();
//    virtual void changeViewer();

    sofa::simulation::Node* getScene()
    {
        if (mViewer) return mViewer->getScene(); else return NULL;
    }

    virtual void unload()
    {
        if ( getScene() )
        {
            mViewer->getPickHandler()->reset();
            mViewer->getPickHandler()->unload();
            mViewer->unloadVisualScene();
            simulation::getSimulation()->unload ( mViewer->getScene() );
            mViewer->setScene(NULL);
        }
    }

    virtual void fileOpen(std::string filename, bool temporaryFile)
    {
        if ( sofa::helper::system::DataRepository.findFile (filename) )
            filename = sofa::helper::system::DataRepository.getFile ( filename );
        else
            return;

        frameCounter = 0;
        sofa::simulation::xml::numDefault = 0;

        this->unloadScene();
        simulation::Node::SPtr root = simulation::getSimulation()->load ( filename.c_str() );
        simulation::getSimulation()->init ( root.get() );
        if ( root == NULL )
        {
            std::cerr<<"Failed to load "<<filename.c_str()<<std::endl;
            return;
        }
        setScene ( root, filename.c_str(), temporaryFile );
        configureGUI(root.get());
    }
protected:
    bool mCreateViewerOpt;// to deal with from RealGUI
    BaseViewer* mViewer;
    const char* mViewerName;

    // remonte createViewer
    // remonte changeViewer => change unloadSceneView with unloadVisualScene
    // remonte SofaMouseManager from initViewer
};


////// TO declare into BaseViewer
//setScene();
//resetView();
//setBackgroundColour(...)
//setBackgroundImage(...)
//setScene()
//getSceneFileName()

} // namespace gui

} // namespace sofa

#endif
