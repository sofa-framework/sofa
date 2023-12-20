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
/** A sample program. Laure Heigeas, Francois Faure, 2007. */
// scene data structure
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/constraint/projective/FixedConstraint.h>
#include <sofa/component/solidmechanics/spring/StiffSpringForceField.h>
#include <sofa/component/mapping/nonlinear/RigidMapping.h>
#include <sofa/component/odesolver/forward/EulerSolver.h>
#include <sofa/component/visual/VisualStyle.h>
#include <sofa/simulation/graph/DAGSimulation.h>
// gui
#include <sofa/gui/common/GUIManager.h>
#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/accessor.h>
#include <sofa/component/init.h>
#include <sofa/simulation/graph/init.h>
#include <sofa/gui/init.h>


typedef sofa::component::odesolver::forward::EulerExplicitSolver Solver;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;

int main(int argc, char** argv)
{
    SOFA_UNUSED(argc);

    //force load all components
    sofa::component::init();
    //force load SofaGui (registering guis)
    sofa::gui::init();

    //To set a specific resolution for the viewer, use the component ViewerSetting in you scene graph
    sofa::gui::common::GUIManager::SetDimension(800, 600);

    sofa::gui::common::GUIManager::Init(argv[0]);
    //=========================== Build the scene
    double endPos = 1.;
    double attach = -1.;
    double splength = 1.;

    //-------------------- The graph root node
    auto groot = sofa::simulation::getSimulation()->createNewGraph("root");
    groot->setGravity({ 0,-10,0 });

    // One solver for all the graph
    auto solver = sofa::core::objectmodel::New<Solver>();
    groot->addObject(solver);
    solver->setName("S");

    //-------------------- Deformable body

    using MechanicalObject3 = sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::Vec3Types>;
    sofa::simulation::Node::SPtr deformableBody = groot.get()->createChild("deformableBody");
    // degrees of freedom
    MechanicalObject3::SPtr DOF = sofa::core::objectmodel::New<MechanicalObject3>();
    deformableBody->addObject(DOF);
    DOF->resize(2);
    DOF->setName("Dof1");

    auto x = sofa::helper::getWriteAccessor(*DOF->write(VecId::position()));
    x[0] = { 0,0,0 };
    x[1] = { endPos,0,0 };

    // mass
    using UniformMass3 = sofa::component::mass::UniformMass<sofa::defaulttype::Vec3Types>;
    auto mass = sofa::core::objectmodel::New<UniformMass3>();
    deformableBody->addObject(mass);
    mass->setMass(1);
    mass->setName("M1");

    // Fixed point
    using FixedConstraint3 = sofa::component::constraint::projective::FixedConstraint<sofa::defaulttype::Vec3Types>;
    auto constraints = sofa::core::objectmodel::New<FixedConstraint3>();
    deformableBody->addObject(constraints);
    constraints->setName("C");
    constraints->addConstraint(0);


    // force field
    using StiffSpringForceField3 = sofa::component::solidmechanics::spring::StiffSpringForceField<sofa::defaulttype::Vec3Types>;
    auto spring = sofa::core::objectmodel::New<StiffSpringForceField3>();
    deformableBody->addObject(spring);
    spring->setName("F1");
    spring->addSpring( 1,0, 100., 1, splength );


    //-------------------- Rigid body
    sofa::simulation::Node::SPtr rigidBody = groot.get()->createChild("rigidBody");

    using MechanicalObjectRigid3 = sofa::component::statecontainer::MechanicalObject<sofa::defaulttype::RigidTypes>;
    // degrees of freedom
    MechanicalObjectRigid3::SPtr rigidDOF = sofa::core::objectmodel::New<MechanicalObjectRigid3>();
    rigidBody->addObject(rigidDOF);
    rigidDOF->resize(1);
    rigidDOF->setName("Dof2");
    auto rigid_x = sofa::helper::getWriteAccessor(*rigidDOF->write(VecId::position()));
    rigid_x[0] = { {endPos - attach + splength,0,0}, sofa::type::Quatd::identity() };

    // mass
    using UniformMassRigid3 = sofa::component::mass::UniformMass<sofa::defaulttype::RigidTypes>;
    UniformMassRigid3::SPtr rigidMass = sofa::core::objectmodel::New<UniformMassRigid3>();
    rigidBody->addObject(rigidMass);
    rigidMass->setName("M2");
    
    auto m = sofa::helper::getWriteAccessor(rigidMass->d_vertexMass);
    m->mass=0.3;
    UniformMassRigid3::MassType::Mat3x3 inertia;
    inertia.fill(0.0);
    float in = 0.1f;
    inertia[0][0] = in;
    inertia[1][1] = in;
    inertia[2][2] = in;
    m->inertiaMatrix = inertia;
    m->recalc();


    //-------------------- the particles attached to the rigid body
    sofa::simulation::Node::SPtr rigidParticles = groot.get()->createChild("rigidParticles");

    // degrees of freedom of the skin
    MechanicalObject3::SPtr rigidParticleDOF = sofa::core::objectmodel::New<MechanicalObject3>();
    rigidParticles->addObject(rigidParticleDOF);
    rigidParticleDOF->resize(1);
    rigidParticleDOF->setName("Dof3");
    auto rp_x = sofa::helper::getWriteAccessor(*rigidParticleDOF->write(VecId::position()));
    rp_x[0] = { attach,0,0 };

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    using RigidMappingRigid3_to_3 = sofa::component::mapping::nonlinear::RigidMapping<sofa::defaulttype::RigidTypes, sofa::defaulttype::Vec3Types>;
    auto rigidMapping = sofa::core::objectmodel::New<RigidMappingRigid3_to_3>();
    rigidMapping->setModels(rigidDOF.get(),rigidParticleDOF.get());

    // Setting paths is redundant with previous line
    // std::string pathobject1("@"+rigidBody->getName()+"/"+rigidDOF->getName());
    // std::string pathobject2("@"+rigidParticles->getName()+"/"+rigidParticleDOF->getName());
    // rigidMapping->setPathInputObject(pathobject1);
    // rigidMapping->setPathOutputObject(pathobject2);

    rigidParticles->addObject( rigidMapping );
    rigidMapping->setName("Map23");


    // ---------------- Interaction force between the deformable and the rigid body
    StiffSpringForceField3::SPtr iff = sofa::core::objectmodel::New<StiffSpringForceField3>( DOF.get(), rigidParticleDOF.get() );
    iff->setPathObject1("@"+deformableBody->getName()+"/"+DOF->getName());
    iff->setPathObject2("@"+rigidParticles->getName()+"/"+rigidParticleDOF->getName());
    groot->addObject(iff);
    iff->setName("F13");
    iff->addSpring( 1,0, 100., 1., splength );



    // Display Flags
    sofa::component::visual::VisualStyle::SPtr style = sofa::core::objectmodel::New<sofa::component::visual::VisualStyle>();
    groot->addObject(style);
    sofa::core::visual::DisplayFlags& flags = *style->displayFlags.beginEdit();
    flags.setShowNormals(false);
    flags.setShowInteractionForceFields(true);
    flags.setShowMechanicalMappings(true);
    flags.setShowCollisionModels(false);
    flags.setShowBoundingCollisionModels(false);
    flags.setShowMappings(false);
    flags.setShowForceFields(true);
    flags.setShowWireFrame(false);
    flags.setShowVisualModels(true);
    flags.setShowBehaviorModels(true);
    style->displayFlags.endEdit();


    //To set a specific resolution for the viewer, use the component ViewerSetting in you scene graph
    sofa::gui::common::GUIManager::SetDimension(800, 600);

    //=========================== Init the scene
    sofa::simulation::node::initRoot(groot.get());
    sofa::gui::common::GUIManager::SetScene(groot);

    groot->setAnimate(true);

    //=========================== Run the main loop
    sofa::gui::common::GUIManager::MainLoop(groot);

    if (groot != NULL)
        sofa::simulation::node::unload(groot);

    sofa::simulation::graph::cleanup();
    return 0;
}
