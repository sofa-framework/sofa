#include <iostream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#ifdef SOFA_GUI_FLTK
#include <sofa/gui/fltk/Main.h>
#endif
#ifdef SOFA_GUI_QT
#include <sofa/gui/qt/Main.h>
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
    //sofa::helper::BackTrace::autodump();
    glutInit(&argc,argv);
    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    std::string gui = "none";
    std::vector<std::string> plugins;
    std::vector<std::string> files;
#ifdef SOFA_GUI_FLTK
    gui = "fltk";
#endif
#ifdef SOFA_GUI_QT
    gui = "qt";
#endif

    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
//	.option(&fileName,'f',"file","scene file")
    .option(&startAnim,'s',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&gui,'g',"gui","choose the UI (none"
#ifdef SOFA_GUI_FLTK
            "|fltk"
#endif
#ifdef SOFA_GUI_QT
            "|qt"
#endif
            ")"
           )
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

    if (!fileName.empty())
    {
        groot = sofa::simulation::tree::Simulation::load(fileName.c_str());
    }
    else
    {
        groot = sofa::simulation::tree::Simulation::load("../Data/demoLiverProximity.scn");
        if (groot == NULL) // Necessary for starting this program under Visual Studio with default Configuration
            //groot = sofa::simulation::tree::Simulation::load("../../../Data/demoLiverProximity.scn");
            groot = sofa::simulation::tree::Simulation::load("../../../Data/demoSphereTree.scn");
    }

    if (groot==NULL)
    {
        groot = new sofa::simulation::tree::GNode;
        //return 1;
    }

    if (startAnim)
        groot->setAnimate(true);

    //=======================================
    // Run the main loop

    if (gui=="none")
    {
        std::cout << "Computing 1000 iterations." << std::endl;
        for (int i=0; i<1000; i++)
        {
            sofa::simulation::tree::Simulation::animate(groot);
        }
        std::cout << "1000 iterations done." << std::endl;
    }
#ifdef SOFA_GUI_FLTK
    else if (gui=="fltk")
    {
        sofa::gui::fltk::MainLoop(argv[0],groot);
    }
#endif
#ifdef SOFA_GUI_QT
    else if (gui=="qt")
    {
        sofa::gui::qt::MainLoop(argv[0],groot,fileName.c_str());
    }
#endif
    else
    {
        std::cerr << "Unsupported GUI."<<std::endl;
        exit(1);
    }

    if (groot!=NULL)
        sofa::simulation::tree::Simulation::unload(groot);
    return 0;
}
