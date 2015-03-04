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
#include <iostream>
#include <sstream>
#include <fstream>

#include <tinyxml.h>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/PluginManager.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#ifdef SOFA_SMP
#include <sofa/simulation/tree/SMPSimulation.h>
#endif
#include <sofa/simulation/tree/TreeSimulation.h>
#include <SofaComponentMain/init.h>
#include <SofaLoader/ReadState.h>
#include <SofaValidation/CompareState.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/gui/BatchGUI.h>  // For the default number of iterations
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/system/atomic.h>
#ifdef SOFA_SMP
#include <athapascan-1>
#endif /* SOFA_SMP */
#ifdef WIN32
#include <windows.h>
#endif
using std::cerr;
using std::endl;

// bool loadConfigurationFile(const std::string& filePath)
// {
//     TiXmlDocument doc;
//     doc.LoadFile();

//     if (!(doc.LoadFile(filePath)))
//     {
//         std::cerr << "Error while loading configuration file: " << filePath << std::endl;
//         return false;
//     }

//     TiXmlElement* root = doc.FirstChildElement("RunSofaConfig");
//     for(TiXmlElement* elt = root->FirstChildElement("ResourcePath");
//         elt != NULL;
//         elt = elt->NextSiblingElement("ResourcePath"))
//     {
//         const std::string path = elt->GetText();
//         sofa::helper::system::DataRepository.addFirstPath(makeAbsolutePath(path));
//     }

//     for(TiXmlElement* elt = root->FirstChildElement("PluginPath");
//         elt != NULL;
//         elt = elt->NextSiblingElement("PluginPath"))
//     {
//         const std::string path = elt->GetText();
//         sofa::helper::system::PluginRepository.addFirstPath(makeAbsolutePath(path));
//     }

//     TiXmlElement* elt = root->FirstChildElement("ScreenshotDirectory");
//     if (elt != NULL)
//     {
//         const std::string path = elt->GetText();
//         sofa::gui::BaseGUI::setScreenshotDirectoryPath(makeAbsolutePath(path));
//     }

//     return true;
// }

void loadVerificationData(std::string& directory, std::string& filename, sofa::simulation::Node* node)
{
    std::cout << "loadVerificationData from " << directory << " and file " << filename << std::endl;

    std::string refFile;

    refFile += directory;
    refFile += '/';
    refFile += sofa::helper::system::SetDirectory::GetFileName(filename.c_str());

    std::cout << "loadVerificationData " << refFile << std::endl;


    sofa::component::misc::CompareStateCreator compareVisitor(sofa::core::ExecParams::defaultInstance());
    compareVisitor.setCreateInMapping(true);
    compareVisitor.setSceneName(refFile);
    compareVisitor.execute(node);

    sofa::component::misc::ReadStateActivator v_read(sofa::core::ExecParams::defaultInstance(), true);
    v_read.execute(node);
}

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    //std::cout << "Using " << sofa::helper::system::atomic<int>::getImplName()<<" atomics." << std::endl;

    sofa::helper::BackTrace::autodump();

    sofa::core::ExecParams::defaultInstance()->setAspectID(0);

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

    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    bool        testMode = false;
    int         nbIterations = sofa::gui::BatchGUI::DEFAULT_NUMBER_OF_ITERATIONS;
    unsigned int nbMSSASamples = 1;
    unsigned    computationTimeSampling=0; ///< Frequency of display of the computation time statistics, in number of animation steps. 0 means never.

    std::string gui = "";
    std::string verif = "";
#ifdef SOFA_SMP
    std::string simulationType = "smp";
#else
    std::string simulationType = "tree";
#endif
    std::vector<std::string> plugins;
    std::vector<std::string> files;
#ifdef SOFA_SMP
    std::string nProcs="";
    bool        disableStealing = false;
    bool        affinity = false;
#endif

    std::string gui_help = "choose the UI (";
    gui_help += sofa::gui::GUIManager::ListSupportedGUI('|');
    gui_help += ")";

    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
    // alphabetical order on short name
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&computationTimeSampling,'c',"computationTimeSampling","Frequency of display of the computation time statistics, in number of animation steps. 0 means never.")
    .option(&gui,'g',"gui",gui_help.c_str())
    .option(&plugins,'l',"load","load given plugins")
    .option(&nbIterations,'n',"nb_iterations","(only batch) Number of iterations of the simulation")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&loadRecent,'r',"recent","load most recently opened file")
    .option(&simulationType,'s',"simu","select the type of simulation (bgl, dag, tree, smp)")
    .option(&temporaryFile,'t',"temporary","the loaded scene won't appear in history of opened files")
    .option(&testMode,'x',"test","select test mode with xml output after N iteration")
    .option(&verif,'v',"verification","load verification data for the scene")
    .option(&nbMSSASamples, 'm', "msaa", "number of samples for MSAA (Multi Sampling Anti Aliasing ; value < 2 means disabled")
#ifdef SOFA_SMP
    .option(&disableStealing,'w',"disableStealing","Disable Work Stealing")
    .option(&nProcs,'c',"nprocs","Number of processor")
    .option(&affinity,'f',"affinity","Enable aFfinity base Work Stealing")
#endif
    (argc,argv);

#ifdef SOFA_SMP
    int ac = 0;
    char **av = NULL;

    Util::KaapiComponentManager::prop["util.globalid"]="0";
    Util::KaapiComponentManager::prop["sched.strategy"]="I";
    if(!disableStealing)
        Util::KaapiComponentManager::prop["sched.stealing"]="true";
    if(nProcs!="")
        Util::KaapiComponentManager::prop["community.thread.poolsize"]=nProcs;
    if(affinity)
    {
        Util::KaapiComponentManager::prop["sched.stealing"]="true";
        Util::KaapiComponentManager::prop["sched.affinity"]="true";
    }

    a1::Community com = a1::System::join_community( ac, av);
#endif /* SOFA_SMP */

#ifndef SOFA_NO_OPENGL
    if(gui!="batch") glutInit(&argc,argv);
#endif

#ifdef SOFA_SMP
        if (simulationType == "smp")
            sofa::simulation::setSimulation(new sofa::simulation::tree::SMPSimulation());
        else
#endif
#ifdef SOFA_HAVE_DAG
    if (simulationType == "tree")
        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
    else
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
#else //SOFA_HAVE_DAG
    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
#endif

    sofa::component::init();
    sofa::simulation::xml::initXml();

    // Add the plugin directory to PluginRepository
#ifdef WIN32
    const std::string pluginDir = "bin";
#else
    const std::string pluginDir = "lib";
#endif
    sofa::helper::system::PluginRepository.addFirstPath(sofa::gui::BaseGUI::getPathPrefix() + "/" + pluginDir);

    // Initialise paths
    sofa::gui::BaseGUI::setConfigDirectoryPath(sofa::gui::BaseGUI::getPathPrefix() + "/config");
    sofa::gui::BaseGUI::setScreenshotDirectoryPath(sofa::gui::BaseGUI::getPathPrefix() + "/screenshots");

    if (!files.empty())
        fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);

    sofa::helper::system::PluginManager::getInstance().init();

    if(gui.compare("batch") == 0 && nbIterations >= 0)
    {
        std::ostringstream oss ;
        oss << "nbIterations=";
        oss << nbIterations;
        sofa::gui::GUIManager::AddGUIOption(oss.str().c_str());
    }

    if(nbMSSASamples > 1)
    {
        std::ostringstream oss ;
        oss << "msaa=";
        oss << nbMSSASamples;
        sofa::gui::GUIManager::AddGUIOption(oss.str().c_str());
    }

    if (int err = sofa::gui::GUIManager::Init(argv[0],gui.c_str()))
        return err;

    if (fileName.empty())
    {
        if (loadRecent) // try to reload the latest scene
        {
            std::string scenes = sofa::gui::BaseGUI::getConfigDirectoryPath() + "/runSofa.ini";
            std::ifstream mrulist(scenes.c_str());
            std::getline(mrulist,fileName);
            mrulist.close();
        }
        else
            fileName = "Demos/caduceus.scn";

        fileName = sofa::helper::system::DataRepository.getFile(fileName);
    }


    if (int err=sofa::gui::GUIManager::createGUI(NULL))
        return err;

    //To set a specific resolution for the viewer, use the component ViewerSetting in you scene graph
    sofa::gui::GUIManager::SetDimension(800,600);

    sofa::simulation::Node::SPtr groot = sofa::core::objectmodel::SPtr_dynamic_cast<sofa::simulation::Node>( sofa::simulation::getSimulation()->load(fileName.c_str()));
    if (groot==NULL)
    {
        groot = sofa::simulation::getSimulation()->createNewGraph("");
    }

    if (!verif.empty())
    {
        loadVerificationData(verif, fileName, groot.get());
    }

    sofa::simulation::getSimulation()->init(groot.get());
    sofa::gui::GUIManager::SetScene(groot,fileName.c_str(), temporaryFile);


    //=======================================
    //Apply Options

    if (startAnim)
        groot->setAnimate(true);
    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        sofa::helper::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    if( computationTimeSampling>0 )
    {
        sofa::helper::AdvancedTimer::setEnabled("Animate", true);
        sofa::helper::AdvancedTimer::setInterval("Animate", computationTimeSampling);
    }

    //=======================================
    // Run the main loop
    if (int err = sofa::gui::GUIManager::MainLoop(groot,fileName.c_str()))
        return err;
    groot = dynamic_cast<sofa::simulation::Node*>( sofa::gui::GUIManager::CurrentSimulation() );

    if (testMode)
    {
        std::string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
        std::cout << "Exporting to XML " << xmlname << std::endl;
        sofa::simulation::getSimulation()->exportXML(groot.get(), xmlname.c_str());
    }

    if (groot!=NULL)
        sofa::simulation::getSimulation()->unload(groot);


    sofa::gui::GUIManager::closeGUI();

    return 0;
}
