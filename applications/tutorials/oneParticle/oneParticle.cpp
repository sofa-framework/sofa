/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/odesolver/forward/EulerSolver.h>
#include <sofa/component/visual/VisualStyle.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/core/VecId.h>
#include <sofa/gui/common/GUIManager.h>

#include <sofa/helper/accessor.h>

#include <sofa/component/init.h>
#include <sofa/gui/init.h>



using sofa::component::odesolver::forward::EulerExplicitSolver;
using namespace sofa::component::collision;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;
using sofa::core::objectmodel::New;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    SOFA_UNUSED(argc);

    //force load SofaComponentAll
    sofa::component::init();
    //force load SofaGui (registering guis)
    sofa::gui::init();

    //To set a specific resolution for the viewer, use the component ViewerSetting in you scene graph
    sofa::gui::common::GUIManager::SetDimension(800, 600);

    sofa::gui::common::GUIManager::Init(argv[0]);

    // The graph root node
    sofa::simulation::Node::SPtr groot = sofa::simulation::getSimulation()->createNewGraph("root");
    groot->setGravity({ 0,-10,0 });

    // One solver for all the graph
    auto solver = sofa::core::objectmodel::New<EulerExplicitSolver>();
    solver->setName("solver");
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // One node to define the particle
    sofa::simulation::Node::SPtr particule_node = groot.get()->createChild("particle_node");
    // The particule, i.e, its degrees of freedom : a point with a velocity
    using MechanicalObject3 = sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types>;
    auto dof = sofa::core::objectmodel::New<MechanicalObject3>();
    dof->setName("particle");
    particule_node->addObject(dof);
    dof->resize(1);
    // get write access the particle positions vector
    auto positions = sofa::helper::getWriteAccessor(*dof->write(VecId::position()));
    positions[0] = { 0,0,0 };
    // get write access the particle velocities vector
    auto velocities = sofa::helper::getWriteAccessor(*dof->write(VecId::velocity()));
    velocities[0] = { 0,0,0 };
    // show the particle
    dof->showObject.setValue(true);
    dof->showObjectScale.setValue(10.);

    // Its properties, i.e, a simple mass node
    using UniformMass3 = sofa::component::mass::UniformMass<sofa::defaulttype::Vec3Types>;
    auto mass = sofa::core::objectmodel::New<UniformMass3>();
    mass->setName("mass");
    particule_node->addObject(mass);
    mass->setMass( 1 );

    // this currently reveals a bug
//    // attach a collision surface to the particle
//    SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphere = New<SphereCollisionModel<sofa::defaulttype::Vec3Types>>();
//    particule_node->addObject(sphere);
//    sphere->defaultRadius.setValue(0.1);

    // Display Flags
    sofa::component::visual::VisualStyle::SPtr style = sofa::core::objectmodel::New<sofa::component::visual::VisualStyle>();
    groot->addObject(style);
    sofa::core::visual::DisplayFlags& flags = *style->displayFlags.beginEdit();
    flags.setShowBehaviorModels(true);
    flags.setShowCollisionModels(true);
    style->displayFlags.endEdit();

    sofa::simulation::node::initRoot(groot.get());
    groot->setAnimate(false);

    //======================================
    // Set up the GUI
    sofa::gui::common::GUIManager::Init(argv[0]);
    sofa::gui::common::GUIManager::createGUI(groot);
    sofa::gui::common::GUIManager::SetDimension(800,700);
//    sofa::gui::common::GUIManager::SetFullScreen();  // why does this not work ?


    //=======================================
    // Run the main loop
    sofa::gui::common::GUIManager::MainLoop(groot);

    sofa::simulation::graph::cleanup();
    return 0;
}
