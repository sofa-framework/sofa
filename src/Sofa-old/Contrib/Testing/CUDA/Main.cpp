#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#include "mycuda.h"

#include "Sofa-old/Components/Graph/Simulation.h"
#include "Sofa-old/Components/Graph/Action.h"
#include "Sofa-old/Components/Common/Factory.h"
#include "Sofa-old/Components/Thread/CTime.h"
#include "Sofa-old/Components/Thread/Automate.h"
#include "Sofa-old/Components/Thread/ThreadSimulation.h"
#include "Sofa-old/Components/Thread/ExecBus.h"
#include "Sofa-old/Components/Thread/Node.h"
#if defined(SOFA_GUI_QT)
#include "Sofa-old/GUI/QT/Main.h"
#elif defined(SOFA_GUI_FLTK)
#include "Sofa-old/GUI/FLTK/Main.h"
#endif

using Sofa::Components::Thread::CTime;
using Sofa::Components::Thread::ctime_t;

using namespace Sofa::Components::Graph;
using namespace Sofa::Contrib::CUDA;

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    //std::string fileName = "../../../../../Data/Benchmarks/GPU/Bar10-spring-rk4-8.scn";
    //std::string fileName = "../../../../../Data/Benchmarks/GPU/Bar10-spring-implicit-8.scn";
    std::string fileName = "../../../../../Data/Benchmarks/GPU/Bar8+4-spring-rk4-4.scn";
    int nbIter = 0;
    if (argc < 2 || argc > 3)
    {
        std::cerr << "Usage: "<<argv[0]<<" filename.scn [niterations]\n";
        //return 1;
    }
    else
    {
        fileName = argv[1];
        if (argc >=3) nbIter = atoi(argv[2]);
    }

    mycudaInit(0);

    GNode* groot = NULL;
    ctime_t t0, t1;
    CTime::getRefTime();

    if (!fileName.empty())
    {
        groot = Simulation::load(fileName.c_str());
    }

    if (groot==NULL)
    {
        groot = new GNode;
    }

    if (nbIter > 0)
    {

        groot->setAnimate(true);

        std::cout << "Computing first iteration." << std::endl;

        Simulation::animate(groot);

        //=======================================
        // Run the main loop

        std::cout << "Computing " << nbIter << " iterations." << std::endl;
        t0 = CTime::getRefTime();

        //=======================================
        // SEQUENTIAL MODE
        int n = 0;
        for (int i=0; i<nbIter; i++)
        {
            int n2 = i*80/(nbIter-1);
            while(n2>n)
            {
                std::cout << '.' << std::flush;
                ++n;
            }
            Simulation::animate(groot);
        }

        t1 = CTime::getRefTime();
        std::cout << std::endl;
        std::cout << nbIter << " iterations done." << std::endl;
        std::cout << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it." << std::endl;
        std::string logname = fileName.substr(0,fileName.length()-4)+"-log.txt";
        std::ofstream flog(logname.c_str());
        flog << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it." << std::endl;
        flog.close();
        std::string objname = fileName.substr(0,fileName.length()-4)+"-scene.obj";
        std::cout << "Exporting to OBJ " << objname << std::endl;
        Simulation::exportOBJ(groot, objname.c_str());
        std::string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
        std::cout << "Exporting to XML " << xmlname << std::endl;
        Simulation::exportXML(groot, xmlname.c_str());

    }
    else
    {
#if defined(SOFA_GUI_QT)
        Sofa::GUI::QT::MainLoop(argv[0],groot,fileName.c_str());
#elif defined(SOFA_GUI_FLTK)
        Sofa::GUI::FLTK::MainLoop(argv[0],groot);
#endif
    }

    return 0;
}
