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
#include <stdarg.h>
#include <stdio.h>

#include <sofa/gpu/cuda/mycuda.h>
#include <sofa/helper/ArgumentParser.h>
#ifdef SOFA_SMP
#include <sofa/simulation/tree/SMPSimulation.h>
#else
#include <sofa/simulation/tree/TreeSimulation.h>
#endif
#include <sofa/simulation/tree/GNode.h>
#include <sofa/component/init.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/helper/system/glut.h>
#ifdef SOFA_SMP
#include <athapascan-1>
#endif /* SOFA_SMP */
#include <sofa/simulation/common/xml/initXml.h>

#ifdef SOFA_DEV
#include <sofa/gpu/cuda/initCudaDev.h>
#include <sofa/component/initMiscMappingDev.h>
#include <sofa/component/initAdvancedFEM.h>
#endif


using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

using namespace sofa::simulation::tree;
using namespace sofa::gpu::cuda;

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
    sofa::helper::BackTrace::autodump();

    //glutInit(&argc,argv);
    sofa::gui::SofaGUI::SetProgramName(argv[0]);
    std::string fileName ;
    bool        startAnim = false;
    bool        printFactory = false;
    bool        loadRecent = false;
    int iter=0;
    int interval=0;
    std::string gui = sofa::gui::SofaGUI::GetGUIName();
    std::vector<std::string> plugins;
    std::vector<std::string> files;
#ifdef SOFA_SMP
    std::string nProcs="";
    bool        disableStealing = false;
    bool        affinity = false;
    std::string cuda="";
    bool	staticGpuPrioritary = false;
    bool	dynamicGpuPrioritary = false;
#endif
    bool silent = false;
    bool verbose = false;
    sofa::gui::SofaGUI::SetProgramName(argv[0]);
    std::string gui_help = "choose the UI (";
    gui_help += sofa::gui::GUIManager::ListSupportedGUI('|');
    gui_help += ")";


    sofa::helper::parse(&files, "This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'S',"start","start the animation loop")
    .option(&printFactory,'p',"factory","print factory logs")
    .option(&gui,'g',"gui",gui_help.c_str())
    .option(&plugins,'l',"load","load given plugins")
    .option(&loadRecent,'r',"recent","load most recently opened file")
    .option(&silent,'s',"silent", "remove most CUDA log messages")
    .option(&interval,'t',"interval", "remove most CUDA log messages")
    .option(&verbose,'v',"verbose","display trace of CUDA calls")
    .option(&iter,'i',"nbiter","Number of iterations")
#ifdef SOFA_SMP
    .option(&disableStealing,'d',"disableStealing","Disable Work Stealing")
    .option(&affinity,'a',"affinity","Enable affinity base Work Stealing")
    .option(&nProcs,'n',"nprocs","Number of processor")
    .option(&cuda,'c',"cuda","Number of CUDA GPU to use")
    .option(&staticGpuPrioritary,'o',"sgpuprio","Static gpu prioritary")
    .option(&dynamicGpuPrioritary,'O',"dgpuprio","Dynamic gpu prioritary")
#endif
    (argc,argv);
    if (silent && verbose)
        mycudaVerboseLevel = LOG_INFO;
    else if (silent)
        mycudaVerboseLevel = LOG_ERR;
    else if (verbose)
        mycudaVerboseLevel = LOG_TRACE;


    //std::cerr<<"mycudaInit(0)"<<"\n";
    //std::cerr<<"apres mycudaInit(0)"<<"\n";
#ifdef SOFA_SMP
    int ac=0;
    char **av=NULL;

    Util::KaapiComponentManager::prop["util.globalid"]="0";
    Util::KaapiComponentManager::prop["sched.strategy"]="I";
    if(!disableStealing)
        Util::KaapiComponentManager::prop["sched.stealing"]="true";
    else
    {
        std::cout << "Work Stealing Disabled!" << std::endl;
        Util::KaapiComponentManager::prop["sched.stealing"]="false";
        Util::KaapiComponentManager::prop["community.cpuset"]="255";
    }
    if(affinity)
    {
        Util::KaapiComponentManager::prop["sched.stealing"]="true";
        Util::KaapiComponentManager::prop["sched.affinity"]="true";
    }

    if(cuda!="")
        Util::KaapiComponentManager::prop["community.thread.gpu"]=cuda;

    if(nProcs!="")
        Util::KaapiComponentManager::prop["community.thread.poolsize"]=nProcs;
    if(staticGpuPrioritary)
        Util::KaapiComponentManager::prop["core.staticgpuprioritary"]="true";

    if(dynamicGpuPrioritary)
    {
        Util::KaapiComponentManager::prop["core.dynamicgpuprioritary"]="true";
        Util::KaapiComponentManager::prop["core.staticgpuprioritary"]="false";
    }
    a1::Community com = a1::System::join_community( ac, av);
    Core::Thread::get_current()->set_cpuset(0);
    Core::Thread::get_current()->set_cpu(0);
    mycudaInit(atoi(nProcs.c_str()));
#endif /* SOFA_SMP*/


    if (!files.empty()) fileName = files[0];
    else fileName =
            "CUDA/quadSpringSphereCUDA.scn"
//             "CUDA/beam10x10x46-spring-rk4-CUDA.scn"
            ;


    int nbIter = iter;

    if (!nbIter)
        glutInit(&argc,argv);


    mycudaInit();

#ifdef SOFA_SMP
    sofa::simulation::setSimulation(new sofa::simulation::tree::SMPSimulation());
#else
    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
#endif
    sofa::component::init();

#ifdef SOFA_DEV
    // load cuda_dev library
    sofa::gpu::cuda::initCudaDev();
    sofa::component::initMiscMappingDev();
    sofa::component::initAdvancedFEM();
#endif

    sofa::simulation::xml::initXml();

    if (!nbIter)
    {
        if (int err=sofa::gui::GUIManager::Init(argv[0],gui.c_str()))
            return err;

        if (int err=sofa::gui::GUIManager::createGUI(NULL))
            return err;

        //To set a specific resolution for the viewer, use the component ViewerDimensionSetting in you scene graph
        //sofa::gui::GUIManager::SetDimension(800,600);
        sofa::gui::GUIManager::SetDimension(640,480);

    }

    sofa::helper::system::DataRepository.findFile(fileName);

    GNode::SPtr groot = NULL;
    ctime_t t0, t1,t2;
    CTime::getRefTime();

    if (!fileName.empty())
    {
        groot = sofa::core::objectmodel::SPtr_dynamic_cast< GNode >(getSimulation()->load(fileName.c_str()));
    }

    if (groot==NULL)
    {
        groot = sofa::core::objectmodel::New<GNode>();
    }
    sofa::simulation::tree::getSimulation()->init(groot.get());
    if (!nbIter)
        sofa::gui::GUIManager::SetScene(groot,fileName.c_str());
#ifdef SOFA_SMP
    Core::Thread::get_current()->set_cpuset(~0UL);
#endif

    if (nbIter != 0)
    {

        groot->setAnimate(true);

        std::cout << "Computing first iteration." << std::endl;

        getSimulation()->animate(groot.get());

        //=======================================
        // Run the main loop
        bool save = (nbIter > 0);
        if (nbIter < 0) nbIter = -nbIter;
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
            t2 = CTime::getRefTime();
            getSimulation()->animate(groot.get());
#ifdef SOFA_SMP
            if(i%20==0)
                mycudaPrintMem();
#endif
//                std::cerr << "All Time: " << ((CTime::getRefTime()-t2)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds " << std::endl;
            if (save)
            {
                if(interval&&i%interval==1)
                {
                    std::ostringstream objname;
                    objname<< fileName.substr(0,fileName.length()-4)<<"-"<<i<<"-scene.obj";
                    std::cout << "Exporting to OBJ " << objname.str() << std::endl;
                    getSimulation()->exportOBJ(groot.get(), objname.str().c_str());
                }
            }
        }

        t1 = CTime::getRefTime();
        std::cout << std::endl;
        std::cout << nbIter << " iterations done." << std::endl;
        std::cout << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it ("<< ((double)nbIter*1000.0)/((t1-t0)/(CTime::getRefTicksPerSec()/1000)) << " FPS)" << std::endl;
        std::string logname = fileName.substr(0,fileName.length()-4)+"-log.txt";
        std::ofstream flog(logname.c_str());
        flog << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it ("<< ((double)nbIter*1000.0)/((t1-t0)/(CTime::getRefTicksPerSec()/1000)) << " FPS)" << std::endl;
        flog.close();
        if (save)
        {
            std::string objname = fileName.substr(0,fileName.length()-4)+"-scene.obj";
            std::cout << "Exporting to OBJ " << objname << std::endl;
            getSimulation()->exportOBJ(groot.get(), objname.c_str());
            std::string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
            std::cout << "Exporting to XML " << xmlname << std::endl;
            getSimulation()->exportXML(groot.get(), xmlname.c_str());
        }

    }
    else
    {
        sofa::gui::GUIManager::MainLoop(groot,fileName.c_str());
        groot = dynamic_cast<GNode*>( sofa::gui::GUIManager::CurrentSimulation() );
    }
    if (groot!=NULL) getSimulation()->unload(groot);
#ifdef SOFA_SMP
    a1::Sync();
#endif

    return 0;
}
