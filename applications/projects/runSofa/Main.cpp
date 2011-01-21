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
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/common/Node.h>


#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/CompareState.h>

#ifdef SOFA_DEV
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#ifdef SOFA_SMP
#include <sofa/simulation/tree/SMPSimulation.h>
#endif
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/component/init.h>
#include <sofa/component/misc/ReadState.h>
#include <sofa/component/misc/CompareState.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/system/atomic.h>
#ifdef SOFA_SMP
#include <athapascan-1>
#endif /* SOFA_SMP */

#ifndef WIN32
#include <dlfcn.h>
bool loadPlugin(const char* filename)
{
    void *handle;
    handle=dlopen(filename, RTLD_LAZY);
    if (!handle)
    {
        std::cerr<<"Error loading plugin "<<filename<<": "<<dlerror()<<std::endl;
        return false;
    }
    std::cerr<<"Plugin "<<filename<<" loaded."<<std::endl;
    return true;
}
#else
bool loadPlugin(const char* filename)
{
    HINSTANCE DLLHandle;
    DLLHandle = LoadLibraryA(filename); //warning: issue between unicode and ansi encoding on Visual c++ -> force to ansi-> dirty!
    if (DLLHandle == NULL)
    {
        std::cerr<<"Error loading plugin "<<filename<<std::endl;
        return false;
    }
    std::cerr<<"Plugin "<<filename<<" loaded."<<std::endl;
    return true;
}
#endif

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

    sofa::component::misc::ReadStateActivator v_read(true, sofa::core::ExecParams::defaultInstance());
    v_read.execute(node);
}

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    //std::cout << "Using " << sofa::helper::system::atomic<int>::getImplName()<<" atomics." << std::endl;

    sofa::helper::BackTrace::autodump();


    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    int			nbIterations = 0;

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
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&gui,'g',"gui",gui_help.c_str())
    .option(&nbIterations,'n',"nb_iterations","(only batch) Number of iterations of the simulation")
    .option(&simulationType,'s',"simu","select the type of simulation (bgl, tree)")
    .option(&plugins,'l',"load","load given plugins")
    .option(&loadRecent,'r',"recent","load most recently opened file")
    .option(&temporaryFile,'t',"temporary","the loaded scene won't appear in history of opened files")
    .option(&verif,'v',"verification","load verification data for the scene")
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

    if(gui!="batch") glutInit(&argc,argv);

#ifdef SOFA_DEV
    if (simulationType == "bgl")
        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    else
#endif
#ifdef SOFA_SMP
        if (simulationType == "smp")
            sofa::simulation::setSimulation(new sofa::simulation::tree::SMPSimulation());
        else
#endif
            sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    sofa::component::init();
    sofa::simulation::xml::initXml();

    if (!files.empty())
        fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());

    if(gui.compare("batch") == 0 && nbIterations > 0)
    {
        std::ostringstream oss ;
        oss << "nbIterations=";
        oss << nbIterations;
        sofa::gui::GUIManager::AddGUIOption(oss.str().c_str());
    }

    if (int err = sofa::gui::GUIManager::Init(argv[0],gui.c_str()))
        return err;

    if (fileName.empty())
    {
        if (loadRecent) // try to reload the latest scene
        {
            std::string scenes = "config/Sofa.ini";
            scenes = sofa::helper::system::DataRepository.getFile( scenes );
            std::ifstream mrulist(scenes.c_str());
            std::getline(mrulist,fileName);
            mrulist.close();
        }
        else
            fileName = "Demos/liver.scn";

        fileName = sofa::helper::system::DataRepository.getFile(fileName);
    }


    if (int err=sofa::gui::GUIManager::createGUI(NULL))
        return err;

    //To set a specific resolution for the viewer, use the component ViewerDimensionSetting in you scene graph
    sofa::gui::GUIManager::SetDimension(800,600);

    sofa::simulation::Node* groot = dynamic_cast<sofa::simulation::Node*>( sofa::simulation::getSimulation()->load(fileName.c_str()));
    if (groot==NULL)
    {
        groot = sofa::simulation::getSimulation()->newNode("");
    }

    if (!verif.empty())
    {
        loadVerificationData(verif, fileName, groot);
    }

    sofa::simulation::getSimulation()->init(groot);
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


    //=======================================
    // Run the main loop
    if (int err = sofa::gui::GUIManager::MainLoop(groot,fileName.c_str()))
        return err;

    groot = dynamic_cast<sofa::simulation::Node*>( sofa::gui::GUIManager::CurrentSimulation() );

    if (groot!=NULL)
        sofa::simulation::getSimulation()->unload(groot);

    return 0;
}
