#include <iostream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/FileRepository.h>
#ifdef SOFA_GUI_FLTK
#include <sofa/gui/fltk/Main.h>
#elif  SOFA_GUI_QTVIEWER
#include <sofa/gui/viewer/Main.h>
#elif  SOFA_GUI_QGLVIEWER
#include <sofa/gui/viewer/Main.h>
#elif  SOFA_GUI_QTOGREVIEWER
#include <sofa/gui/viewer/Main.h>
#endif

#include <GL/glut.h>

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
    std::cerr << "Plugin loading not supported on this platform.\n";
    return false;
}
#endif

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

#ifndef SOFA_GUI_QTOGREVIEWER
    glutInit(&argc,argv);
#endif

    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    bool nogui = false;
    std::string gui = "none";
    std::vector<std::string> plugins;
    std::vector<std::string> files;

    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
//	.option(&fileName,'f',"file","scene file")
    .option(&startAnim,'s',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&nogui,'g',"nogui","use no gui, run a number of iterations and exit")
    .option(&plugins,'l',"load","load given plugins")
    (argc,argv);

    if (!files.empty()) fileName = files[0];

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());

    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        sofa::helper::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    sofa::simulation::tree::GNode* groot = NULL;

    if (fileName.empty())
    {
        fileName = "liver.scn";
        sofa::helper::system::DataRepository.findFile(fileName);
    }

    //=======================================
    // Run the main loop

    if (nogui)
    {
        groot = sofa::simulation::tree::Simulation::load(fileName.c_str());

        if (groot==NULL)
        {
            std::cerr<<"Could not load file "<<fileName<<std::endl;
            return 1;
        }
        std::cout << "Computing 1000 iterations." << std::endl;
        for (int i=0; i<1000; i++)
        {
            sofa::simulation::tree::Simulation::animate(groot);
        }
        std::cout << "1000 iterations done." << std::endl;
        return 0;
    }
#ifdef SOFA_GUI_FLTK
    else if (gui=="fltk")
    {
        sofa::gui::fltk::MainLoop(argv[0],groot);
    }
//#else
#endif
#ifdef SOFA_GUI_QTVIEWER
    sofa::gui::guiviewer::MainLoop(argv[0],groot,fileName.c_str());
    // BUGFIX: the user may have loaded another simulation, in which case the first simulation is already destroyed
    // So we need to get the current simulation from the GUI
    groot = sofa::gui::guiviewer::CurrentSimulation();
#endif
#ifdef SOFA_GUI_QGLVIEWER
    sofa::gui::guiviewer::MainLoop(argv[0],groot,fileName.c_str());
    // BUGFIX: the user may have loaded another simulation, in which case the first simulation is already destroyed
    // So we need to get the current simulation from the GUI
    groot = sofa::gui::guiviewer::CurrentSimulation();
#endif
#ifdef SOFA_GUI_QTOGREVIEWER

    sofa::gui::guiviewer::MainLoop(argv[0],groot,fileName.c_str());
    // BUGFIX: the user may have loaded another simulation, in which case the first simulation is already destroyed
    // So we need to get the current simulation from the GUI
    groot = sofa::gui::guiviewer::CurrentSimulation();
#endif


    if (groot!=NULL)
        sofa::simulation::tree::Simulation::unload(groot);
    return 0;
}
