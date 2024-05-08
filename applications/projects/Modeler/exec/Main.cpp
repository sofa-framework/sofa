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
#include "../lib/SofaModeler.h"

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <sofa/simulation/common/init.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/gui/common/GUIManager.h>

#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/helper/logging/Messaging.h>

#include <SofaComponentAll/initSofaComponentAll.h>

#include <QApplication>

#include <iostream>
#include <fstream>
#include <sofa/gui/common/GuiDataRepository.h>
#include <sofa/helper/logging/MessageDispatcher.h>
#include <sofa/helper/logging/ConsoleMessageHandler.h>
#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>
#include <sofa/gui/common/BaseGUI.h>
#include <sofa/gui/batch/init.h>

using sofa::gui::common::BaseGUI;
using sofa::helper::logging::MainPerComponentLoggingMessageHandler;
using sofa::helper::logging::ConsoleMessageHandler;
using sofa::helper::logging::MessageDispatcher;
using sofa::gui::common::GuiDataRepository;
using sofa::helper::system::FileSystem;
using sofa::helper::Utils;
using sofa::gui::common::GUIManager;
using sofa::core::ExecParams ;
using sofa::simulation::graph::DAGSimulation;
using sofa::helper::system::SetDirectory;
using sofa::core::objectmodel::BaseNode ;
using sofa::helper::system::DataRepository;
using sofa::helper::system::PluginRepository;
using sofa::helper::system::PluginManager;
using namespace std;


// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{

    // Add resources dir to GuiDataRepository
    const std::string runSofaIniFilePath = Utils::getSofaPathTo("/etc/runSofa.ini");
    std::map<std::string, std::string> iniFileValues = Utils::readBasicIniFile(runSofaIniFilePath);
    if (iniFileValues.find("RESOURCES_DIR") != iniFileValues.end())
    {
        std::string dir = iniFileValues["RESOURCES_DIR"];
        dir = SetDirectory::GetRelativeFromProcess(dir.c_str());
        if(FileSystem::isDirectory(dir))
        {
            sofa::gui::common::GuiDataRepository.addFirstPath(dir);
        }
    }

#if defined(SOFA_HAVE_DAG)
    string simulationType = "dag";
#else
    string simulationType = "tree";
#endif

    vector<string> plugins;
    vector<string> files;

    string gui_help = "choose the UI (";
    gui_help += GUIManager::ListSupportedGUI('|');
    gui_help += ")";

    sofa::simulation::common::init();
    sofa::simulation::graph::init();

    MessageDispatcher::clearHandlers() ;
    MessageDispatcher::addHandler( new ConsoleMessageHandler() ) ;
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;
#ifdef TRACY_ENABLE
    MessageDispatcher::addHandler(&sofa::helper::logging::MainTracyMessageHandler::getInstance());
#endif

    // Output FileRepositories
    msg_info("Modeler") << "PluginRepository paths = " << PluginRepository.getPathsJoined();
    msg_info("Modeler") << "DataRepository paths = " << DataRepository.getPathsJoined();
    msg_info("Modeler") << "GuiDataRepository paths = " << GuiDataRepository.getPathsJoined();

    // Initialise paths
    BaseGUI::setConfigDirectoryPath(Utils::getSofaPathPrefix() + "/config", true);
    BaseGUI::setScreenshotDirectoryPath(Utils::getSofaPathPrefix() + "/screenshots", true);

    // Add Batch GUI (runSofa without any GUIs wont be useful)
    sofa::gui::batch::init();

    for (unsigned int i=0; i<plugins.size(); i++)
        PluginManager::getInstance().loadPlugin(plugins[i]);

    std::string configPluginPath = sofa_tostring(CONFIG_PLUGIN_FILENAME);
    std::string defaultConfigPluginPath = sofa_tostring(DEFAULT_CONFIG_PLUGIN_FILENAME);

    if (PluginRepository.findFile(configPluginPath, "", nullptr))
    {
        msg_info("Modeler") << "Loading automatically plugin list in " << configPluginPath;
        PluginManager::getInstance().readFromIniFile(configPluginPath);
    }
    else if (PluginRepository.findFile(defaultConfigPluginPath, "", nullptr))
    {
        msg_info("Modeler") << "Loading automatically plugin list in " << defaultConfigPluginPath;
        PluginManager::getInstance().readFromIniFile(defaultConfigPluginPath);
    }
    else
    {
        msg_info("Modeler") << "No plugin list found. No plugin will be automatically loaded.";
    }

    PluginManager::getInstance().init();


    QApplication* application = new QApplication(argc, argv);
    (void)application;

    sofa::gui::qt::SofaModeler* sofaModeler = new sofa::gui::qt::SofaModeler();

    sofaModeler->show();

    std::string binaryName=argv[0];
#ifdef WIN32
    const std::string exe=".exe";
    if (binaryName.size() > exe.size()) binaryName = binaryName.substr(0, binaryName.size()-exe.size());
#endif
    if (!binaryName.empty() && binaryName[binaryName.size()-1] == 'd') sofaModeler->setDebugBinary(true);

    QString pathIcon=(sofa::helper::system::DataRepository.getFirstPath() + std::string( "/icons/MODELER.png" )).c_str();
    application->setWindowIcon(QIcon(pathIcon));

    for (int i=1; i<argc; ++i)
    {
        //Try to open the simulations passed in command line
        sofaModeler->fileOpen(std::string(argv[i]));
    }
    if (argc <= 1 ) sofaModeler->newTab();

    int appReturnCode = application->exec();
    delete application;
    return appReturnCode;
}
