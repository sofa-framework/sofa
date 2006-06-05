#include <iostream>
#include <fstream>
#include "argumentParser.h"
#include "Sofa/Components/Graph/Simulation.h"
#include "Sofa/Components/Common/Factory.h"
#ifdef SOFA_GUI_FLTK
#include "Sofa/GUI/FLTK/Main.h"
#endif
#ifdef SOFA_GUI_QT
#include "Sofa/GUI/QT/Main.h"
#endif

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    std::string fileName = "Data/demo6Triangle.scn";
    bool        startAnim = false;
    bool        printFactory = false;
    std::string gui = "none";
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
    (argc,argv);

    if (printFactory)
    {
        std::cout << "////////// FACTORY //////////" << std::endl;
        Sofa::Components::Common::printFactoryLog();
        std::cout << "//////// END FACTORY ////////" << std::endl;
    }

    Sofa::Components::Graph::GNode* groot = NULL;

    groot = Sofa::Components::Graph::Simulation::load(fileName.c_str());

    if (groot==NULL)
        return 1;

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
        Sofa::GUI::QT::MainLoop(argv[0],groot);
    }
#endif
    else
    {
        std::cerr << "Unsupported GUI."<<std::endl;
        exit(1);
    }
    return 0;
}
