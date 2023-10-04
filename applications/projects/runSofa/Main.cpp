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
#include <sstream>
using std::ostringstream ;
#include <fstream>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <runSofaValidation.h>

#include <sofa/simulation/Node.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/config.h> // #defines SOFA_HAVE_DAG (or not)
#include <sofa/simulation/common/init.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
using sofa::simulation::Node;
#include <sofa/simulation/SceneLoaderFactory.h>
#include <SceneChecking/SceneCheckerListener.h>
using sofa::scenechecking::SceneCheckerListener;

#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/cast.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/Utils.h>
#include <sofa/gui/common/GUIManager.h>
using sofa::gui::common::GUIManager;

using sofa::core::ExecParams ;

#include <sofa/helper/system/console.h>
using sofa::helper::Utils;

using sofa::simulation::graph::DAGSimulation;
using sofa::helper::system::SetDirectory;
using sofa::core::objectmodel::BaseNode ;

#include <sofa/gui/common/BaseGUI.h>
using sofa::gui::common::BaseGUI;

#include <sofa/gui/batch/init.h>

#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::ConsoleMessageHandler ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using  sofa::helper::logging::RichConsoleStyleMessageFormatter ;

#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>
using  sofa::helper::logging::MainPerComponentLoggingMessageHandler ;

#include <sofa/helper/AdvancedTimer.h>

#include <sofa/gui/common/GuiDataRepository.h>
using sofa::gui::common::GuiDataRepository ;

using sofa::helper::system::DataRepository;
using sofa::helper::system::PluginRepository;
using sofa::helper::system::PluginManager;

#include <sofa/helper/logging/MessageDispatcher.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <sofa/helper/logging/ExceptionMessageHandler.h>
using sofa::helper::logging::ExceptionMessageHandler;

#ifdef TRACY_ENABLE
#include <sofa/helper/logging/TracyMessageHandler.h>
#endif

#include <sofa/gui/common/ArgumentParser.h>


void addGUIParameters(sofa::gui::common::ArgumentParser* argumentParser)
{
    GUIManager::RegisterParameters(argumentParser);
}

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

    sofa::helper::BackTrace::autodump();

#ifdef WIN32
    {
        HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
        COORD s;
        s.X = 160; s.Y = 10000;
        SetConsoleScreenBufferSize(hStdout, s);
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(hStdout, &csbi))
        {
            SMALL_RECT winfo;
            winfo = csbi.srWindow;
            //winfo.Top = 0;
            winfo.Left = 0;
            //winfo.Bottom = csbi.dwSize.Y-1;
            winfo.Right = csbi.dwMaximumWindowSize.X-1;
            SetConsoleWindowInfo(hStdout, TRUE, &winfo);
        }

    }
#endif

    string fileName ;
    bool        startAnim = false;
    bool        showHelp = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    bool        testMode = false;
    bool        noAutoloadPlugins = false;
    bool        noSceneCheck = false;
    unsigned int nbMSSASamples = 1;
    bool computationTimeAtBegin = false;
    unsigned int computationTimeSampling=0; ///< Frequency of display of the computation time statistics, in number of animation steps. 0 means never.
    string    computationTimeOutputType="stdout";

    string gui = "";
    string verif = "";

#if defined(SOFA_HAVE_DAG)
    string simulationType = "dag";
#else
    string simulationType = "tree";
#endif

    vector<string> plugins;
    vector<string> files;

    string colorsStatus = "unset";
    string messageHandler = "auto";
    bool enableInteraction = false ;
    int width = 800;
    int height = 600;

    string gui_help = "choose the UI (";
    gui_help += GUIManager::ListSupportedGUI('|');
    gui_help += ")";

    // Argument parser has 2 stages
    // one is for the runSofa options itself
    // second is for the eventual options the GUIs can add (i.e batch with the "-n" number of iterations option) 
    sofa::gui::common::ArgumentParser* argParser = new sofa::gui::common::ArgumentParser(argc, argv);

    argParser->addArgument(
        cxxopts::value<bool>(showHelp)
        ->default_value("false")
        ->implicit_value("true"),
        "h,help",
        "Display this help message"
    );

    argParser->addArgument(
        cxxopts::value<bool>(startAnim)
        ->default_value("false")
        ->implicit_value("true"),
        "a,start",
        "start the animation loop"
    );
    argParser->addArgument(
        cxxopts::value<bool>(computationTimeAtBegin)
        ->default_value("false")
        ->implicit_value("true"),
        "b,computationTimeAtBegin",
        "Output computation time statistics of the init (at the begin of the simulation)"
    );
    argParser->addArgument(
        cxxopts::value<unsigned int>(computationTimeSampling)
        ->default_value("0"),
        "computationTimeSampling",
        "Frequency of display of the computation time statistics, in number of animation steps. 0 means never."
    );
    argParser->addArgument(
        cxxopts::value<std::string>(computationTimeOutputType)
        ->default_value("stdout"),
        "o,computationTimeOutputType",
        "Output type for the computation time statistics: either stdout, json or ljson"
    );
    argParser->addArgument(
        cxxopts::value<std::string>(gui)->default_value(""),
        "g,gui",
        gui_help.c_str()
    );
    argParser->addArgument(
        cxxopts::value<std::vector<std::string>>(plugins),
        "l,load",
        "load given plugins"
    );
    argParser->addArgument(
        cxxopts::value<bool>(noAutoloadPlugins)
        ->default_value("false")
        ->implicit_value("true"),
        "noautoload",
        "disable plugins autoloading"
    );
    argParser->addArgument(
        cxxopts::value<bool>(noSceneCheck)
        ->default_value("false")
        ->implicit_value("true"),
        "noscenecheck",
        "disable scene checking for each scene loading"
    );
    argParser->addArgument(
        cxxopts::value<bool>(printFactory)
        ->default_value("false")
        ->implicit_value("true"),
        "p,factory",
        "print factory logs"
    );
    argParser->addArgument(
        cxxopts::value<bool>(loadRecent)
        ->default_value("false")->implicit_value("true"),
        "r,recent",
        "load most recently opened file"
    );
    argParser->addArgument(
        cxxopts::value<std::string>(simulationType),
        "s,simu", 
        "select the type of simulation (bgl, dag, tree)"
    );
    argParser->addArgument(
        cxxopts::value<bool>(temporaryFile)
        ->default_value("false")->implicit_value("true"),
        "tmp",
        "the loaded scene won't appear in history of opened files"
    );
    argParser->addArgument(
        cxxopts::value<bool>(testMode)
        ->default_value("false")->implicit_value("true"),
        "test",
        "select test mode with xml output after N iteration"
    );
    argParser->addArgument(
        cxxopts::value<std::string>(verif)
        ->default_value(""),
        "v,verification",
        "load verification data for the scene"
    );
    argParser->addArgument(
        cxxopts::value<std::string>(colorsStatus)
        ->default_value("auto")
        ->implicit_value("yes"),
        "c,colors",
        "use colors on stdout and stderr (yes, no, auto)"
    );
    argParser->addArgument(
        cxxopts::value<std::string>(messageHandler)
        ->default_value("auto"),
        "f,formatting",
        "select the message formatting to use (auto, clang, sofa, rich, test)"
    );
    argParser->addArgument(
        cxxopts::value<bool>(enableInteraction)
        ->default_value("false")
        ->implicit_value("true"),
        "i,interactive",
        "enable interactive mode for the GUI which includes idle and mouse events (EXPERIMENTAL)"
    );
    argParser->addArgument(
        cxxopts::value<std::vector<std::string> >(),
        "argv",
        "forward extra args to the python interpreter"
    );
    argParser->addArgument(
        cxxopts::value<unsigned int>(nbMSSASamples)
        ->default_value("1"),
        "msaa",
        "Number of samples for MSAA (Multi Sampling Anti Aliasing ; value < 2 means disabled"
    );

    // first option parsing to see if the user requested to show help
    argParser->parse();

    if(showHelp)
    {
        argParser->showHelp();
        exit( EXIT_SUCCESS );
    }

    // Note that initializations must be done after ArgumentParser that can exit the application (without cleanup)
    // even if everything is ok e.g. asking for help
    sofa::simulation::graph::init();

    if (simulationType == "tree")
        msg_warning("runSofa") << "Tree based simulation, switching back to graph simulation.";
    assert(sofa::simulation::getSimulation());

    if (colorsStatus == "unset") {
        // If the parameter is unset, check the environment variable
        const char * colorStatusEnvironment = std::getenv("SOFA_COLOR_TERMINAL");
        if (colorStatusEnvironment != nullptr) {
            const std::string status (colorStatusEnvironment);
            if (status == "yes" || status == "on" || status == "always")
                sofa::helper::console::setStatus(sofa::helper::console::Status::On);
            else if (status == "no" || status == "off" || status == "never")
                sofa::helper::console::setStatus(sofa::helper::console::Status::Off);
            else
                sofa::helper::console::setStatus(sofa::helper::console::Status::Auto);
        }
    } else if (colorsStatus == "auto")
        sofa::helper::console::setStatus(sofa::helper::console::Status::Auto);
    else if (colorsStatus == "yes")
        sofa::helper::console::setStatus(sofa::helper::console::Status::On);
    else if (colorsStatus == "no")
        sofa::helper::console::setStatus(sofa::helper::console::Status::Off);

    //TODO(dmarchal): Use smart pointer there to avoid memory leaks !!
    if (messageHandler == "auto" )
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ConsoleMessageHandler() ) ;
    }
    else if (messageHandler == "clang")
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ClangMessageHandler() ) ;
    }
    else if (messageHandler == "sofa")
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ConsoleMessageHandler() ) ;
    }
    else if (messageHandler == "rich")
    {
        MessageDispatcher::clearHandlers() ;
        MessageDispatcher::addHandler( new ConsoleMessageHandler(&RichConsoleStyleMessageFormatter::getInstance()) ) ;
    }
    else if (messageHandler == "test"){
        MessageDispatcher::addHandler( new ExceptionMessageHandler() ) ;
    }
    else{
        msg_warning("") << "Invalid argument '" << messageHandler << "' for '--formatting'";
    }
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;
#ifdef TRACY_ENABLE
    MessageDispatcher::addHandler(&sofa::helper::logging::MainTracyMessageHandler::getInstance());
#endif

    // Output FileRepositories
    msg_info("runSofa") << "PluginRepository paths = " << PluginRepository.getPathsJoined();
    msg_info("runSofa") << "DataRepository paths = " << DataRepository.getPathsJoined();
    msg_info("runSofa") << "GuiDataRepository paths = " << GuiDataRepository.getPathsJoined();

    // Initialise paths
    BaseGUI::setConfigDirectoryPath(Utils::getSofaPathPrefix() + "/config", true);
    BaseGUI::setScreenshotDirectoryPath(Utils::getSofaPathPrefix() + "/screenshots", true);

    // Add Batch GUI (runSofa without any GUIs wont be useful)
    sofa::gui::batch::init();

    for (unsigned int i=0; i<plugins.size(); i++)
        PluginManager::getInstance().loadPlugin(plugins[i]);

    if (!noAutoloadPlugins)
    {
        std::string configPluginPath = sofa_tostring(CONFIG_PLUGIN_FILENAME);
        std::string defaultConfigPluginPath = sofa_tostring(DEFAULT_CONFIG_PLUGIN_FILENAME);

        if (PluginRepository.findFile(configPluginPath, "", nullptr))
        {
            msg_info("runSofa") << "Loading automatically plugin list in " << configPluginPath;
            PluginManager::getInstance().readFromIniFile(configPluginPath);
        }
        else if (PluginRepository.findFile(defaultConfigPluginPath, "", nullptr))
        {
            msg_info("runSofa") << "Loading automatically plugin list in " << defaultConfigPluginPath;
            PluginManager::getInstance().readFromIniFile(defaultConfigPluginPath);
        }
        else
        {
            msg_info("runSofa") << "No plugin list found. No plugin will be automatically loaded.";
        }
    }
    else
    {
        msg_info("runSofa") << "Automatic plugin loading disabled.";
    }

    // Parse again to take into account the potential new options
    addGUIParameters(argParser);
    argParser->parse();

    // Fetching file name must be done after the additionnal potential options have been added
    // otherwise the first parsing will take the unknown options as the file name
    // (because of its positional parameter)
    files = argParser->getInputFileList();

    PluginManager::getInstance().init();

    if (int err = GUIManager::Init(argv[0],gui.c_str()))
    {
        sofa::simulation::common::cleanup();
        sofa::simulation::graph::cleanup();

        return err;
    }

    if (!files.empty())
        fileName = files[0];

    if (fileName.empty())
    {
        if (loadRecent) // try to reload the latest scene
        {
            string scenes = BaseGUI::getConfigDirectoryPath() + "/runSofa.ini";
            std::ifstream mrulist(scenes.c_str());
            std::getline(mrulist,fileName);
            mrulist.close();
        }
        else
            fileName = "Demos/caduceus.scn";

        fileName = DataRepository.getFile(fileName);
    }

    if (int err=GUIManager::createGUI(nullptr))
        return err;

    //To set a specific resolution for the viewer, use the component ViewerSetting in you scene graph
    GUIManager::SetDimension(width, height);
    GUIManager::CenterWindow();

    // Create and register the SceneCheckerListener before scene loading
    if(!noSceneCheck)
    {
        sofa::simulation::SceneLoader::addListener( SceneCheckerListener::getInstance() );
    }

    const std::vector<std::string> sceneArgs = sofa::gui::common::ArgumentParser::extra_args();
    Node::SPtr groot = sofa::simulation::node::load(fileName, false, sceneArgs);
    if( !groot )
        groot = sofa::simulation::getSimulation()->createNewGraph("");

    if (!verif.empty())
    {
        runSofa::Validation::execute(verif, fileName, groot.get());
    }

    if( computationTimeAtBegin )
    {
        sofa::helper::AdvancedTimer::setEnabled("Init", true);
        sofa::helper::AdvancedTimer::setInterval("Init", 1);
        sofa::helper::AdvancedTimer::setOutputType("Init", computationTimeOutputType);
        sofa::helper::AdvancedTimer::begin("Init");
    }

    sofa::simulation::node::initRoot(groot.get());
    if( computationTimeAtBegin )
    {
        msg_info("") << sofa::helper::AdvancedTimer::end("Init", groot->getTime(), groot->getDt());
    }

    //=======================================
    //Apply Options

    // start anim option
    if (startAnim)
        groot->setAnimate(true);

    // set scene and animation root to the gui
    GUIManager::SetScene(groot, fileName.c_str(), temporaryFile);

    if (printFactory)
    {
        msg_info("") << "////////// FACTORY //////////" ;
        sofa::helper::printFactoryLog();
        msg_info("") << "//////// END FACTORY ////////" ;
    }

    if( computationTimeSampling>0 )
    {
        sofa::helper::AdvancedTimer::setEnabled("Animate", true);
        sofa::helper::AdvancedTimer::setInterval("Animate", computationTimeSampling);
        sofa::helper::AdvancedTimer::setOutputType("Animate", computationTimeOutputType);
    }

    //=======================================
    // Run the main loop
    if (int err = GUIManager::MainLoop(groot,fileName.c_str()))
        return err;
    groot = dynamic_cast<Node*>( GUIManager::CurrentSimulation() );

    if (testMode)
    {
        string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
        msg_info("") << "Exporting to XML " << xmlname ;
        sofa::simulation::node::exportInXML(groot.get(), xmlname.c_str());
    }

    if (groot!=nullptr)
        sofa::simulation::node::unload(groot);


    GUIManager::closeGUI();

    sofa::simulation::common::cleanup();
    sofa::simulation::graph::cleanup();
    return 0;
}
