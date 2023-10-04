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
#include <sofa/gui/common/BaseGUI.h>
#include <sofa/gui/common/BaseViewer.h>

#include <sofa/type/vector.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileSystem.h>

#include <sofa/core/objectmodel/ConfigurationSetting.h>
#include <sofa/component/setting/ViewerSetting.h>
#include <sofa/component/setting/MouseButtonSetting.h>
#include <sofa/component/setting/BackgroundSetting.h>
#include <sofa/component/setting/StatsSetting.h>
#include <sofa/component/setting/SofaDefaultPathSetting.h>

#include <algorithm>
#include <cstring>

#include <sofa/simulation/ExportGnuplotVisitor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

using namespace sofa::simulation;
using sofa::helper::system::FileSystem;
using sofa::helper::Utils;

using namespace sofa::simulation;
namespace sofa::gui::common
{

const char* BaseGUI::mProgramName = nullptr;
std::string BaseGUI::mGuiName = "";
std::string BaseGUI::configDirectoryPath = ".";
std::string BaseGUI::screenshotDirectoryPath = ".";
ArgumentParser* BaseGUI::mArgumentParser = nullptr;

BaseGUI::BaseGUI()
{

}

BaseGUI::~BaseGUI()
{

}

void BaseGUI::configureGUI(sofa::simulation::Node::SPtr groot)
{

    sofa::component::setting::SofaDefaultPathSetting *defaultPath;
    groot->get(defaultPath, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (defaultPath)
    {
        if (!defaultPath->gnuplotPath.getValue().empty())
            setGnuplotPath(defaultPath->gnuplotPath.getValue());
    }


    //Background
    sofa::component::setting::BackgroundSetting *background;
    groot->get(background, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (background)
    {
        if (background->image.getValue().empty())
            setBackgroundColor(background->color.getValue());
        else
            setBackgroundImage(background->image.getFullPath());
    }

    //Stats
    sofa::component::setting::StatsSetting *stats;
    groot->get(stats, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (stats)
    {
        setDumpState(stats->dumpState.getValue());
        setLogTime(stats->logTime.getValue());
        setExportState(stats->exportState.getValue());
#ifdef SOFA_DUMP_VISITOR_INFO
        setTraceVisitors(stats->traceVisitors.getValue());
#endif
    }

    //Viewer Dimension TODO in viewer !
    sofa::component::setting::ViewerSetting *viewerConf;
    groot->get(viewerConf, sofa::core::objectmodel::BaseContext::SearchRoot);
    if (viewerConf) setViewerConfiguration(viewerConf);

    //TODO: Video Recorder Configuration

    //Mouse Manager using ConfigurationSetting component...
    sofa::type::vector< sofa::component::setting::MouseButtonSetting*> mouseConfiguration;
    groot->get<sofa::component::setting::MouseButtonSetting>(&mouseConfiguration, sofa::core::objectmodel::BaseContext::SearchRoot);

    for (unsigned int i=0; i<mouseConfiguration.size(); ++i)  setMouseButtonConfiguration(mouseConfiguration[i]);

}

void BaseGUI::exportGnuplot(sofa::simulation::Node* node, std::string /*gnuplot_directory*/ )
{
    SCOPED_TIMER_VARNAME(exportGnuplotTimer, "exportGnuplot");

    const sofa::core::ExecParams* params = sofa::core::execparams::defaultInstance();
    ExportGnuplotVisitor expg ( params, node->getTime());
    node->execute ( expg );
}

bool BaseGUI::saveScreenshot(const std::string& filename, int compression_level)
{
    if(getViewer())
    {
        getViewer()->screenshot(filename, compression_level);
        return true;
    }
    else return false;
}

const std::string& BaseGUI::getConfigDirectoryPath()
{
    return configDirectoryPath;
}

const std::string& BaseGUI::getScreenshotDirectoryPath()
{
    return screenshotDirectoryPath;
}

static void setDirectoryPath(std::string& outputVariable, const std::string& path, bool createIfNecessary)
{
    const bool pathExists = FileSystem::exists(path);

    if (!pathExists && !createIfNecessary)
    {
        msg_error("BaseGUI") << "No such directory '" << path << "'";
    }
    else if (pathExists && !FileSystem::isDirectory(path))
    {
         msg_error("BaseGUI") << "Not a directory: " << path << "'";
    }
    else
    {
        if (!pathExists)
        {
            FileSystem::createDirectory(path);
            msg_info("BaseGUI") << "Created directory: " << path;
        }
        outputVariable = path;
    }
}

void BaseGUI::setConfigDirectoryPath(const std::string& path, bool createIfNecessary)
{
    setDirectoryPath(configDirectoryPath, path, createIfNecessary);
}

void BaseGUI::setScreenshotDirectoryPath(const std::string& path, bool createIfNecessary)
{
    setDirectoryPath(screenshotDirectoryPath, path, createIfNecessary);
}


} // namespace sofa::gui::common
