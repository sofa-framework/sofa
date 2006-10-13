#include <iostream>
#include <fstream>
#include "argumentParser.h"
#include "Sofa/Components/Graph/Simulation.h"
#include "Sofa/Components/Common/Factory.h"
#include "Sofa/Components/Common/BackTrace.h"
#ifdef SOFA_GUI_FLTK
#include "Sofa/GUI/FLTK/Main.h"
#endif
#ifdef SOFA_GUI_QT
#include "Sofa/GUI/QT/Main.h"
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
    //Sofa::Components::Common::BackTrace::autodump();
    glutInit(&argc,argv);
    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    std::string gui = "none";
    std::vector<std::string> plugins;
#ifdef SOFA_GUI_FLTK
    gui = "fltk";
#endif
#ifdef SOFA_GUI_QT
    gui = "qt";
#endif

    parse("This is a SOFA application. Here are the command line arguments")
    .option(&fileName,'f',"file","scene file")
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

    for (unsigned int i=0; i<plugins.size(); i++)
        loadPlugin(plugins[i].c_str());

    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        Sofa::Components::Common::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    Sofa::Components::Graph::GNode* groot = NULL;

    if (!fileName.empty())
    {
        groot = Sofa::Components::Graph::Simulation::load(fileName.c_str());
    }
    else
    {
        groot = Sofa::Components::Graph::Simulation::load("../Data/demoLiverProximity.scn");
        if (groot == NULL) // Necessary for starting this program under Visual Studio with default Configuration
            groot = Sofa::Components::Graph::Simulation::load("../../../Data/demoLiverProximity.scn");
    }

    if (groot==NULL)
    {
        groot = new Sofa::Components::Graph::GNode;
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
            Sofa::Components::Graph::Simulation::animate(groot);
        }
        std::cout << "1000 iterations done." << std::endl;
    }
#ifdef SOFA_GUI_FLTK
    else if (gui=="fltk")
    {
        Sofa::GUI::FLTK::MainLoop(argv[0],groot);
    }
#endif
#ifdef SOFA_GUI_QT
    else if (gui=="qt")
    {
        Sofa::GUI::QT::MainLoop(argv[0],groot,fileName.c_str());

    }
#endif
    else
    {
        std::cerr << "Unsupported GUI."<<std::endl;
        exit(1);
    }

    if (groot!=NULL)
        Sofa::Components::Graph::Simulation::unload(groot);
    return 0;
}
