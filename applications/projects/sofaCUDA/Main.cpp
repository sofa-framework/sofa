#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#include <sofa/gpu/cuda/mycuda.h>

#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/init.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>
#include <sofa/helper/system/glut.h>

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

using namespace sofa::simulation::tree;
using namespace sofa::gpu::cuda;

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{

    glutInit(&argc,argv);

    //std::string fileName = "CUDA/beam10x10x46-spring-rk4-CUDA.scn";

    std::string fileName = "CUDA/quadSpringSphereCUDA.scn";

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

    sofa::component::init();

    sofa::gui::SofaGUI::Init(argv[0]);

    sofa::helper::system::DataRepository.findFile(fileName);

    mycudaInit();

    GNode* groot = NULL;
    ctime_t t0, t1;
    CTime::getRefTime();

    if (!fileName.empty())
    {
        groot = dynamic_cast< GNode* >(getSimulation()->load(fileName.c_str()));
    }

    if (groot==NULL)
    {
        groot = new GNode;
    }

    if (nbIter > 0)
    {

        groot->setAnimate(true);

        std::cout << "Computing first iteration." << std::endl;

        getSimulation()->animate(groot);

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
            getSimulation()->animate(groot);
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
        getSimulation()->exportOBJ(groot, objname.c_str());
        std::string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
        std::cout << "Exporting to XML " << xmlname << std::endl;
        getSimulation()->exportXML(groot, xmlname.c_str());

    }
    else
    {
        sofa::gui::SofaGUI::MainLoop(groot,fileName.c_str());
    }

    return 0;
}
