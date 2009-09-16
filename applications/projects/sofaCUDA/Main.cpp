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
#include <fstream>
#include <stdarg.h>
#include <stdio.h>

#include <sofa/gpu/cuda/mycuda.h>

#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/simulation/tree/GNode.h>
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
        sofa::simulation::tree::getSimulation()->init(groot);
    }

    if (groot==NULL)
    {
        groot = new GNode;
    }

    if (nbIter != 0)
    {

        groot->setAnimate(true);

        std::cout << "Computing first iteration." << std::endl;

        getSimulation()->animate(groot);

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
        if (save)
        {
            std::string objname = fileName.substr(0,fileName.length()-4)+"-scene.obj";
            std::cout << "Exporting to OBJ " << objname << std::endl;
            getSimulation()->exportOBJ(groot, objname.c_str());
            std::string xmlname = fileName.substr(0,fileName.length()-4)+"-scene.scn";
            std::cout << "Exporting to XML " << xmlname << std::endl;
            getSimulation()->exportXML(groot, xmlname.c_str());
        }

    }
    else
    {
        sofa::gui::SofaGUI::MainLoop(groot,fileName.c_str());
    }

    return 0;
}
