/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/gui/common/config.h>

#include <sofa/type/RGBAColor.h>
#include <sofa/type/Vec.h>
#include <sofa/simulation/fwd.h>

#include <sofa/component/setting/ViewerSetting.h>
#include <sofa/component/setting/MouseButtonSetting.h>

namespace sofa::gui::common
{

class BaseViewer;
class ArgumentParser;

class SOFA_GUI_COMMON_API BaseGUI
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
    virtual void setScene(sofa::simulation::NodeSPtr groot, const char* filename=nullptr, bool temporaryFile=false)=0;
    /// Get the rootNode of the sofa scene
    virtual sofa::simulation::Node* currentSimulation() = 0;
    /// @}

    /// Use a component setting to configure our GUI
    virtual void configureGUI(sofa::simulation::NodeSPtr groot);

    /// @name methods to configure the GUI
    /// @{
    virtual void setDumpState(bool) {}
    virtual void setLogTime(bool) {}
    virtual void setExportState(bool) {}
#ifdef SOFA_DUMP_VISITOR_INFO
    virtual void setTraceVisitors(bool) {}
#endif
    virtual void setGnuplotPath(const std::string & /*path*/) {}

    virtual void initViewer(BaseViewer* /*viewer*/) {}
    virtual void setViewerConfiguration(sofa::component::setting::ViewerSetting* /*viewerConf*/) {}
    virtual void setViewerResolution(int /* width */, int /* height */) {}
    virtual void setFullScreen() {}
    virtual void centerWindow() {}
    virtual void setBackgroundColor(const sofa::type::RGBAColor& /*color*/) {}
    virtual void setBackgroundImage(const std::string& /*image*/) {}

    virtual BaseViewer* getViewer() {return nullptr;}
    virtual void registerViewer(BaseViewer* /*viewer*/) {}
    virtual bool saveScreenshot(const std::string& filename, int compression_level =-1);

    virtual void setMouseButtonConfiguration(sofa::component::setting::MouseButtonSetting* /*button*/) {}
    /// @}

    /// @name methods to communicate with the GUI
    /// @{
    /// Do one step of the GUI loop
    virtual void stepMainLoop() {}
    /// Send a (script) message
    virtual void sendMessage(const std::string & /*msgType*/,const std::string & /*msgValue*/) {}
    /// Force the displayed FPS value (if any)
    virtual void showFPS(double /*fps*/) {}
    /// @}

    void exportGnuplot(sofa::simulation::Node* node, std::string gnuplot_directory="");

    static std::string& GetGUIName() { return mGuiName; }

    static const char* GetProgramName() { return mProgramName; }
    static void SetProgramName(const char* argv0) { if(argv0) mProgramName = argv0;}
    static void SetArgumentParser(ArgumentParser* parser) {mArgumentParser = parser;}

    static const std::string& getConfigDirectoryPath();
    static const std::string& getScreenshotDirectoryPath();
    static void setConfigDirectoryPath(const std::string& path, bool createIfNecessary = false);
    static void setScreenshotDirectoryPath(const std::string& path, bool createIfNecessary = false);

    /// If the function returns true: when the GUI is created, its name will be saved so that it will be created when
    /// no GUI is specified. If the function returns false, the GUI name is not saved, and the last one will be used
    /// when no GUI is specified.
    virtual bool canBeDefaultGUI() const { return true; }

protected:
    BaseGUI();
    /// The destructor should not be called directly. Use the closeGUI() method instead.
    virtual ~BaseGUI();

    static std::string mGuiName;
    static std::string configDirectoryPath;
    static std::string screenshotDirectoryPath;
    static const char* mProgramName;
    static ArgumentParser* mArgumentParser;
};

} // namespace sofa::gui::common
