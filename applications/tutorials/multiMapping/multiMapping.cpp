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
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#ifdef SOFA_HAVE_BGL
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/init.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
#include "../../../applications/tutorials/objectCreator/ObjectCreator.h"


#include <sofa/simulation/common/Simulation.h>
#include <tutorials/objectCreator/ObjectCreator.h>

using namespace sofa;
using namespace sofa::helper;
using helper::vector;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using namespace sofa::component::container;
using namespace sofa::component::topology;
using namespace sofa::component::collision;
using namespace sofa::component::visualmodel;
using namespace sofa::component::mapping;
using namespace sofa::component::forcefield;

typedef SReal Scalar;
typedef Vec<3,SReal> Vec3;
typedef Vec<1,SReal> Vec1;
typedef component::odesolver::EulerImplicitSolver EulerImplicitSolver;
typedef component::linearsolver::CGLinearSolver<component::linearsolver::GraphScatteredMatrix, component::linearsolver::GraphScatteredVector> CGLinearSolver;


bool startAnim = true;
bool verbose = false;
std::string simulationType = "bgl";
SReal complianceValue = 0.1;
SReal dampingRatio = 0.1;
Vec3 gravity(0,-1,0);
SReal dt = 0.01;



int main(int argc, char** argv)
{

    sofa::helper::BackTrace::autodump();
    sofa::core::ExecParams::defaultInstance()->setAspectID(0);

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&startAnim,'a',"start","start the animation loop")
    .option(&simulationType,'s',"simu","select the type of simulation (bgl, tree)")
    .option(&verbose,'v',"verbose","print debug info")
    (argc,argv);

    glutInit(&argc,argv);

//#ifdef SOFA_DEV
//    if (simulationType == "bgl")
//        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
//    else
//#endif
//        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
#ifdef SOFA_HAVE_DAG
    sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());
#endif

    sofa::component::init();
#ifdef SOFA_GPU_CUDA
#ifdef WIN32
#ifdef NDEBUG
    std::string name("sofagpucuda_1_0.dll");
#else
    std::string name("sofagpucuda_1_0d.dll");
#endif
    sofa::helper::system::DynamicLibrary::load(name);
#endif
#endif

    sofa::gui::initMain();
    if (int err = sofa::gui::GUIManager::Init(argv[0],"")) return err;
    if (int err=sofa::gui::GUIManager::createGUI(NULL)) return err;
    sofa::gui::GUIManager::SetDimension(800,600);

    //=================================================
    sofa::simulation::Node::SPtr groot = SimpleObjectCreator::createGridScene(Vec3(0,0,0), Vec3(5,1,1), 6,2,2, 1.0, 100 );
    //=================================================

    sofa::simulation::getSimulation()->init(groot.get());
    sofa::gui::GUIManager::SetScene(groot);


    // Run the main loop
    if (int err = sofa::gui::GUIManager::MainLoop(groot))
        return err;

    sofa::simulation::getSimulation()->unload(groot);
    sofa::gui::GUIManager::closeGUI();

    return 0;
}



