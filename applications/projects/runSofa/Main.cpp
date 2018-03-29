/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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


#include <sofa/helper/ArgumentParser.h>
#include <SofaSimulationCommon/common.h>
#include <sofa/simulation/Node.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/simulation/config.h> // #defines SOFA_HAVE_DAG (or not)
#include <SofaSimulationCommon/init.h>
#ifdef SOFA_HAVE_DAG
#include <SofaSimulationGraph/init.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#endif
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
using sofa::simulation::Node;

#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <SofaGeneralLoader/ReadState.h>
#include <SofaValidation/CompareState.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/cast.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/Utils.h>
#include <sofa/gui/GUIManager.h>
using sofa::gui::GUIManager;

#include <sofa/gui/Main.h>
#include <sofa/gui/BatchGUI.h>  // For the default number of iterations
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/atomic.h>

using sofa::core::ExecParams ;

#include <sofa/helper/system/console.h>
using sofa::helper::Utils;
using sofa::helper::Console;

using sofa::component::misc::CompareStateCreator;
using sofa::component::misc::ReadStateActivator;
using sofa::simulation::tree::TreeSimulation;
using sofa::simulation::graph::DAGSimulation;
using sofa::helper::system::SetDirectory;
using sofa::core::objectmodel::BaseNode ;
using sofa::gui::BatchGUI;
using sofa::gui::BaseGUI;

#include <sofa/helper/logging/Messaging.h>

#include <sofa/helper/logging/ConsoleMessageHandler.h>
using sofa::helper::logging::ConsoleMessageHandler ;

#include <sofa/core/logging/RichConsoleStyleMessageFormatter.h>
using  sofa::helper::logging::RichConsoleStyleMessageFormatter ;

#include <sofa/core/logging/PerComponentLoggingMessageHandler.h>
using  sofa::helper::logging::MainPerComponentLoggingMessageHandler ;

#ifdef WIN32
#include <windows.h>
#endif

#include <sofa/gui/GuiDataRepository.h>
using sofa::gui::GuiDataRepository ;

using sofa::helper::system::DataRepository;
using sofa::helper::system::PluginRepository;
using sofa::helper::system::PluginManager;

#include <sofa/helper/logging/MessageDispatcher.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <sofa/helper/logging/ExceptionMessageHandler.h>
using sofa::helper::logging::ExceptionMessageHandler;

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)


void loadVerificationData(string& directory, string& filename, Node* node)
{
    msg_info("") << "loadVerificationData from " << directory << " and file " << filename ;

    string refFile;

    refFile += directory;
    refFile += '/';
    refFile += SetDirectory::GetFileName(filename.c_str());

    msg_info("") << "loadVerificationData " << refFile ;

    CompareStateCreator compareVisitor(ExecParams::defaultInstance());
    compareVisitor.setCreateInMapping(true);
    compareVisitor.setSceneName(refFile);
    compareVisitor.execute(node);

    ReadStateActivator v_read(ExecParams::defaultInstance(), true);
    v_read.execute(node);
}

void addGUIParameters(ArgumentParser* argumentParser)
{
    GUIManager::RegisterParameters(argumentParser);
}

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    GuiDataRepository.addFirstPath(Utils::getSofaPathTo("share/sofa/gui/runSofa/resources").c_str()) ;
    sofa::helper::BackTrace::autodump();

    ExecParams::defaultInstance()->setAspectID(0);

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

    sofa::gui::initMain();

    string fileName ;
    bool        startAnim = false;
    bool        showHelp = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    bool        testMode = false;
    bool        noAutoloadPlugins = false;
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
#ifdef SOFA_SMP
    string nProcs="";
    bool        disableStealing = false;
    bool        affinity = false;
#endif
    string colorsStatus = "auto";
    string messageHandler = "auto";
    bool enableInteraction = false ;
    int width = 800;
    int height = 600;

    string gui_help = "choose the UI (";
    gui_help += GUIManager::ListSupportedGUI('|');
    gui_help += ")";

    ArgumentParser* argParser = new ArgumentParser(argc, argv);
    argParser->addArgument(po::value<bool>(&showHelp)->default_value(false)->implicit_value(true),                  "help,h", "Display this help message");
    argParser->addArgument(po::value<bool>(&startAnim)->default_value(false)->implicit_value(true),                 "start,a", "start the animation loop");
    argParser->addArgument(po::value<bool>(&computationTimeAtBegin)->default_value(false)->implicit_value(true),    "computationTimeAtBegin,b", "Output computation time statistics of the init (at the begin of the simulation)");
    argParser->addArgument(po::value<unsigned int>(&computationTimeSampling)->default_value(0),                     "computationTimeSampling", "Frequency of display of the computation time statistics, in number of animation steps. 0 means never.");
    argParser->addArgument(po::value<std::string>(&computationTimeOutputType)->default_value("stdout"),             "computationTimeOutputType,o", "Output type for the computation time statistics: either stdout, json or ljson");
    argParser->addArgument(po::value<std::string>(&gui)->default_value(""),                                         "gui,g", gui_help.c_str());
    argParser->addArgument(po::value<std::vector<std::string>>(&plugins),                                           "load,l", "load given plugins");
    argParser->addArgument(po::value<bool>(&noAutoloadPlugins)->default_value(false)->implicit_value(true),         "noautoload", "disable plugins autoloading");

    // example of an option using lambda function which ensure the value passed is > 0
    argParser->addArgument(po::value<unsigned int>(&nbMSSASamples)->default_value(1)->notifier([](unsigned int value)
    {
        if (value < 1) {
            std::cerr << "msaa sample cannot be lower than 1" << std::endl;
            exit( EXIT_FAILURE );
        }
    }),                                                                                                             "msaa,m", "number of samples for MSAA (Multi Sampling Anti Aliasing ; value < 2 means disabled");

    argParser->addArgument(po::value<bool>(&printFactory)->default_value(false)->implicit_value(true),              "factory,p", "print factory logs");
    argParser->addArgument(po::value<bool>(&loadRecent)->default_value(false)->implicit_value(true),                "recent,r", "load most recently opened file");
    argParser->addArgument(po::value<std::string>(&simulationType),                                                 "simu,s", "select the type of simulation (bgl, dag, tree)");
    argParser->addArgument(po::value<bool>(&temporaryFile)->default_value(false)->implicit_value(true),             "tmp", "the loaded scene won't appear in history of opened files");
    argParser->addArgument(po::value<bool>(&testMode)->default_value(false)->implicit_value(true),                  "test", "select test mode with xml output after N iteration");
    argParser->addArgument(po::value<std::string>(&verif)->default_value(""), "verification,v",                     "load verification data for the scene");
    argParser->addArgument(po::value<std::string>(&colorsStatus)->default_value("auto")->implicit_value("yes"),     "colors,c", "use colors on stdout and stderr (yes, no, auto)");
    argParser->addArgument(po::value<std::string>(&messageHandler)->default_value("auto"), "formatting,f",          "select the message formatting to use (auto, clang, sofa, rich, test)");
    argParser->addArgument(po::value<bool>(&enableInteraction)->default_value(false)->implicit_value(true),         "interactive,i", "enable interactive mode for the GUI which includes idle and mouse events (EXPERIMENTAL)");
    argParser->addArgument(po::value<std::vector<std::string> >()->multitoken(), "argv",                            "forward extra args to the python interpreter");

#ifdef SOFA_SMP
    argParser->addArgument(po::value<bool>(&disableStealing)->default_value(false)->implicit_value(true),           "disableStealing,w", "Disable Work Stealing")
    argParser->addArgument(po::value<std::string>(&nProcs)->default_value(""),                                      "nprocs", "Number of processor")
    argParser->addArgument(po::value<bool>(&affinity)->default_value(false)->implicit_value(true),                  "affinity", "Enable aFfinity base Work Stealing")
#endif

    addGUIParameters(argParser);
    argParser->parse();
    files = argParser->getInputFileList();

    if(showHelp)
    {
        argParser->showHelp();
        exit( EXIT_SUCCESS );
    }

    // Note that initializations must be done after ArgumentParser that can exit the application (without cleanup)
    // even if everything is ok e.g. asking for help
    sofa::simulation::tree::init();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::init();
#endif
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

#ifdef SOFA_HAVE_DAG
    if (simulationType == "tree")
        sofa::simulation::setSimulation(new TreeSimulation());
    else
        sofa::simulation::setSimulation(new DAGSimulation());
#else //SOFA_HAVE_DAG
    sofa::simulation::setSimulation(new TreeSimulation());
#endif

    if (colorsStatus == "auto")
        Console::setColorsStatus(Console::ColorsAuto);
    else if (colorsStatus == "yes")
        Console::setColorsStatus(Console::ColorsEnabled);
    else if (colorsStatus == "no")
        Console::setColorsStatus(Console::ColorsDisabled);

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
        MessageDispatcher::addHandler( new ConsoleMessageHandler(new RichConsoleStyleMessageFormatter()) ) ;
    }
    else if (messageHandler == "test"){
        MessageDispatcher::addHandler( new ExceptionMessageHandler() ) ;
    }
    else{
        msg_warning("") << "Invalid argument '" << messageHandler << "' for '--formatting'";
    }
    MessageDispatcher::addHandler(&MainPerComponentLoggingMessageHandler::getInstance()) ;


    // Add the plugin directory to PluginRepository
    const std::string& pluginDir = Utils::getPluginDirectory();
    PluginRepository.addFirstPath(pluginDir);

    // Initialise paths
    BaseGUI::setConfigDirectoryPath(Utils::getSofaPathPrefix() + "/config", true);
    BaseGUI::setScreenshotDirectoryPath(Utils::getSofaPathPrefix() + "/screenshots", true);

    if (!files.empty())
        fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        PluginManager::getInstance().loadPlugin(plugins[i]);

    std::string configPluginPath = pluginDir + "/" + TOSTRING(CONFIG_PLUGIN_FILENAME);
    std::string defaultConfigPluginPath = pluginDir + "/" + TOSTRING(DEFAULT_CONFIG_PLUGIN_FILENAME);

    if (!noAutoloadPlugins)
    {
        if (DataRepository.findFile(configPluginPath))
        {
            msg_info("runSofa") << "Loading automatically plugin list in " << configPluginPath;
            PluginManager::getInstance().readFromIniFile(configPluginPath);
        }
        else if (DataRepository.findFile(defaultConfigPluginPath))
        {
            msg_info("runSofa") << "Loading automatically plugin list in " << defaultConfigPluginPath;
            PluginManager::getInstance().readFromIniFile(defaultConfigPluginPath);
        }
        else
            msg_info("runSofa") << "No plugin list found. No plugin will be automatically loaded.";
    }
    else
        msg_info("runSofa") << "Automatic plugin loading disabled.";

    PluginManager::getInstance().init();

    if (int err = GUIManager::Init(argv[0],gui.c_str()))
        return err;

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


    if (int err=GUIManager::createGUI(NULL))
        return err;

    //To set a specific resolution for the viewer, use the component ViewerSetting in you scene graph
    GUIManager::SetDimension(width, height);

    Node::SPtr groot = sofa::simulation::getSimulation()->load(fileName.c_str());
    if( !groot )
        groot = sofa::simulation::getSimulation()->createNewGraph("");

    if (!verif.empty())
    {
        loadVerificationData(verif, fileName, groot.get());
    }

    if( computationTimeAtBegin )
    {
        sofa::helper::AdvancedTimer::setEnabled("Init", true);
        sofa::helper::AdvancedTimer::setInterval("Init", 1);
        sofa::helper::AdvancedTimer::setOutputType("Init", computationTimeOutputType);
        sofa::helper::AdvancedTimer::begin("Init");
    }

    sofa::simulation::getSimulation()->init(groot.get());
    if( computationTimeAtBegin )
    {
        msg_info("") << sofa::helper::AdvancedTimer::end("Init", groot.get());
    }
    GUIManager::SetScene(groot,fileName.c_str(), temporaryFile);


    //=======================================
    //Apply Options

    if (startAnim)
        groot->setAnimate(true);
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
        sofa::simulation::getSimulation()->exportXML(groot.get(), xmlname.c_str());
    }

    if (groot!=NULL)
        sofa::simulation::getSimulation()->unload(groot);


    GUIManager::closeGUI();

    sofa::simulation::common::cleanup();
    sofa::simulation::tree::cleanup();
#ifdef SOFA_HAVE_DAG
    sofa::simulation::graph::cleanup();
#endif
    return 0;
}
