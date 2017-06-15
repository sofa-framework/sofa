/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#include <SofaOpenCL/myopencl.h>

#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaSimulationTree/GNode.h>
#include <SofaSimulationTree/init.h>
#include <SofaComponentMain/init.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/glut.h>

using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;

using namespace sofa::simulation::tree;
using namespace sofa::gpu::opencl;

// ---------------------------------------------------------------------
// --- MAIN
// ---------------------------------------------------------------------

int main(int argc, char** argv)
{
    sofa::simulation::tree::init();
    sofa::helper::BackTrace::autodump();
    sofa::component::init();
    sofa::gui::initMain();

    /*sofa::gui::SofaGUI::SetProgramName(argv[0]);
    std::string gui = sofa::gui::SofaGUI::GetGUIName();*/
	std::string gui = "";
    sofa::gui::GUIManager::ListSupportedGUI('|');
    //std::string fileName = "OPENCL/beam10x10x46-spring-rk4-OPENCL.scn";

    std::string fileName = "OPENCL/quadSpringSphereOPENCL.scn";

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

    if (!nbIter)
        glutInit(&argc,argv);

    myopenclInit();

    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());

    if (!nbIter)
    {
        if (int err=sofa::gui::GUIManager::Init(argv[0],gui.c_str()))
            return err;

        if (int err=sofa::gui::GUIManager::createGUI(NULL))
            return err;
    }

    sofa::helper::system::DataRepository.findFile(fileName);

    GNode::SPtr groot = NULL;
    ctime_t t0, t1;
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
            getSimulation()->animate(groot.get());
        }

        t1 = CTime::getRefTime();
        std::cout << std::endl;
        std::cout << nbIter << " iterations done." << std::endl;
        std::cout << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it." << std::endl;
        std::string logname = fileName.substr(0,fileName.length()-4)+"-log.txt";
        std::ofstream flog(logname.c_str());
        flog << "Time: " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))*0.001 << " seconds, " << ((t1-t0)/(CTime::getRefTicksPerSec()/1000))/(double)nbIter <<" ms/it." << std::endl;
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

    sofa::simulation::tree::cleanup();
    return 0;
}
