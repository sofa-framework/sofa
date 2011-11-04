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
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/component/init.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glut.h>
#include <sofa/helper/system/atomic.h>

#include <sofa/core/ObjectFactory.h>


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
bool loadPlugin(const char* /*filename*/)
{
    std::cerr << "Plugin loading not supported on this platform.\n";
    return false;
}
#endif

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::cout << "Using " << sofa::helper::system::atomic<int>::getImplName()<<" atomics." << std::endl;

    sofa::helper::BackTrace::autodump();



    sofa::gui::SofaGUI::SetProgramName(argv[0]);
    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    bool        temporaryFile = false;
    std::string dimension="800x600";
    bool fullScreen = false;

    std::string gui = sofa::gui::SofaGUI::GetGUIName();
    std::vector<std::string> plugins;
    std::vector<std::string> files;

    std::string gui_help = "choose the UI (";
    gui_help += sofa::gui::GUIManager::ListSupportedGUI('|');
    gui_help += ")";

    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'s',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&gui,'g',"gui",gui_help.c_str())
    .option(&plugins,'l',"load","load given plugins")
    .option(&loadRecent,'r',"recent","load most recently opened file")
    .option(&dimension,'d',"dimension","width and height of the viewer")
    .option(&fullScreen,'f',"fullScreen","start in full screen")
    .option(&temporaryFile,'t',"temporary","the loaded scene won't appear in history of opened files")
    (argc,argv);

    if(gui!="batch") glutInit(&argc,argv);

    sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    sofa::component::init();
    sofa::simulation::xml::initXml();



    if (!files.empty()) fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());


    if (int err=sofa::gui::GUIManager::Init(argv[0],gui.c_str()))
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


    sofa::simulation::bgl::BglGraphManager::getInstance()->clear();
    sofa::simulation::bgl::BglNode* groot = dynamic_cast<sofa::simulation::bgl::BglNode*>( sofa::simulation::bgl::getSimulation()->load(fileName.c_str()));
    if (groot==NULL)  groot = new sofa::simulation::bgl::BglNode;

    sofa::simulation::bgl::getSimulation()->init(groot);
    sofa::gui::GUIManager::SetScene(groot,fileName.c_str(), temporaryFile);


    //=======================================
    //Apply Options

    if (startAnim)  groot->setAnimate(true);

    //Dimension Option
    std::string::size_type separator=dimension.find_first_of('x');
    if (separator != std::string::npos)
    {
        std::string stringWidth=dimension.substr(0,separator);
        std::string stringHeight=dimension.substr(separator+1);
        sofa::gui::GUIManager::SetDimension(atoi(stringWidth.c_str()), atoi(stringHeight.c_str()));
    }

    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        sofa::helper::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    if (fullScreen) sofa::gui::GUIManager::SetFullScreen();

    //=======================================
    // Run the main loop
    if (int err=sofa::gui::GUIManager::MainLoop(groot,fileName.c_str()))
        return err;
    groot = dynamic_cast<sofa::simulation::bgl::BglNode*>( sofa::gui::GUIManager::CurrentSimulation() );


    if (groot!=NULL) sofa::simulation::bgl::getSimulation()->unload(groot);
    return 0;
}
