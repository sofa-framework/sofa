/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
/** A sample program. Laure Heigeas, Francois Faure, 2007. */
// scene data structure
#include <SofaSimulationTree/GNode.h>
#include <sofa/simulation/Simulation.h>
#include <SofaSimulationTree/init.h>
#include <SofaSimulationTree/TreeSimulation.h>
#include <SofaGraphComponent/Gravity.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaImplicitOdeSolver/StaticSolver.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaBaseVisual/VisualStyle.h>
// gui
#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/accessor.h>

#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

#include <sofa/component/typedef/Sofa_typedef.h>

using namespace sofa::simulation::tree;
typedef sofa::component::odesolver::EulerSolver Solver;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;

int main(int, char** argv)
{
    sofa::simulation::tree::init();
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();
    sofa::gui::initMain();
    sofa::gui::GUIManager::Init(argv[0]);
    //=========================== Build the scene
    double endPos = 1.;
    double attach = -1.;
    double splength = 1.;

    //-------------------- The graph root node
    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
    sofa::simulation::Node::SPtr groot = sofa::simulation::getSimulation()->createNewGraph("root");
    groot->setGravity( Coord3(0,-10,0) );

    // One solver for all the graph
    Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    groot->addObject(solver);
    solver->setName("S");

    //-------------------- Deformable body

    sofa::simulation::Node::SPtr deformableBody = groot.get()->createChild("deformableBody");
    // degrees of freedom
    MechanicalObject3::SPtr DOF = sofa::core::objectmodel::New<MechanicalObject3>();
    deformableBody->addObject(DOF);
    DOF->resize(2);
    DOF->setName("Dof1");
    WriteAccessor< Data< VecCoord3 > > x = *DOF->write(VecId::position() );
    x[0] = Coord3(0,0,0);
    x[1] = Coord3(endPos,0,0);

    // mass
    //    ParticleMasses* mass = new ParticleMasses;
    UniformMass3::SPtr mass = sofa::core::objectmodel::New<UniformMass3>();
    deformableBody->addObject(mass);
    mass->setMass(1);
    mass->setName("M1");

    // Fixed point
    FixedConstraint3::SPtr constraints = sofa::core::objectmodel::New<FixedConstraint3>();
    deformableBody->addObject(constraints);
    constraints->setName("C");
    constraints->addConstraint(0);


    // force field
    StiffSpringForceField3::SPtr spring = sofa::core::objectmodel::New<StiffSpringForceField3>();
    deformableBody->addObject(spring);
    spring->setName("F1");
    spring->addSpring( 1,0, 100., 1, splength );


    //-------------------- Rigid body
    sofa::simulation::Node::SPtr rigidBody = groot.get()->createChild("rigidBody");

    // degrees of freedom
    MechanicalObjectRigid3::SPtr rigidDOF = sofa::core::objectmodel::New<MechanicalObjectRigid3>();
    rigidBody->addObject(rigidDOF);
    rigidDOF->resize(1);
    rigidDOF->setName("Dof2");
    WriteAccessor< Data<VecCoordRigid3> > rigid_x = *rigidDOF->write(VecId::position() );
    rigid_x[0] = CoordRigid3( Coord3(endPos-attach+splength,0,0),
            Quat3::identity() );

    // mass
    UniformMassRigid3::SPtr rigidMass = sofa::core::objectmodel::New<UniformMassRigid3>();
    rigidBody->addObject(rigidMass);
    rigidMass->setName("M2");
    UniformMassRigid3::MassType* m = rigidMass->d_mass.beginEdit();
    m->mass=0.3;
    UniformMassRigid3::MassType::Mat3x3 inertia;
    inertia.fill(0.0);
    float in = 0.1f;
    inertia[0][0] = in;
    inertia[1][1] = in;
    inertia[2][2] = in;
    m->inertiaMatrix = inertia;
    m->recalc();
    rigidMass->d_mass.endEdit();


    //-------------------- the particles attached to the rigid body
    sofa::simulation::Node::SPtr rigidParticles = groot.get()->createChild("rigidParticles");

    // degrees of freedom of the skin
    MechanicalObject3::SPtr rigidParticleDOF = sofa::core::objectmodel::New<MechanicalObject3>();
    rigidParticles->addObject(rigidParticleDOF);
    rigidParticleDOF->resize(1);
    rigidParticleDOF->setName("Dof3");
    WriteAccessor< Data< VecCoord3 > > rp_x = *rigidParticleDOF->write(VecId::position() );
    rp_x[0] = Coord3(attach,0,0);

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    RigidMappingRigid3_to_3::SPtr rigidMapping = sofa::core::objectmodel::New<RigidMappingRigid3_to_3>();
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
    sofa::component::visualmodel::VisualStyle::SPtr style = sofa::core::objectmodel::New<sofa::component::visualmodel::VisualStyle>();
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

    //=========================== Init the scene
    sofa::simulation::tree::getSimulation()->init(groot.get());
    /*    groot->setAnimate(false);
    */

#ifdef PS3
    groot->setAnimate(true);
#endif


    //=========================== Run the main loop

    sofa::gui::GUIManager::MainLoop(groot);

    sofa::simulation::tree::cleanup();
    return 0;
}

