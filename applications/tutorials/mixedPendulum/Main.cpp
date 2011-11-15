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
/** A sample program. Laure Heigeas, Francois Faure, 2007. */
// scene data structure
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/visualmodel/OglModel.h>
// gui
#include <sofa/gui/GUIManager.h>
#include <sofa/core/VecId.h>
#include <sofa/core/objectmodel/Data.h>
#include <sofa/helper/accessor.h>

#include <sofa/component/typedef/Sofa_typedef.h>

using namespace sofa::simulation::tree;
typedef sofa::component::odesolver::EulerSolver Solver;
using sofa::core::objectmodel::Data;
using sofa::helper::ReadAccessor;
using sofa::helper::WriteAccessor;
using sofa::core::VecId;

int main(int, char** argv)
{
    sofa::gui::GUIManager::Init(argv[0]);
    //=========================== Build the scene
    double endPos = 1.;
    double attach = -1.;
    double splength = 1.;

    //-------------------- The graph root node
    GNode::SPtr groot = sofa::core::objectmodel::New<GNode>();
    groot->setName( "root" );
    groot->setGravity( Coord3(0,-10,0) );

    // One solver for all the graph
    Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();
    groot->addObject(solver);
    solver->setName("S");

    //-------------------- Deformable body
    GNode::SPtr deformableBody = sofa::core::objectmodel::New<GNode>("deformableBody", groot.get());

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
    GNode::SPtr rigidBody = sofa::core::objectmodel::New<GNode>("rigidBody",groot.get());

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
    UniformMassRigid3::MassType* m = rigidMass->mass.beginEdit();
    m->mass=0.3;
    UniformMassRigid3::MassType::Mat3x3 inertia;
    inertia.fill(0.0);
    float in = 0.1f;
    inertia[0][0] = in;
    inertia[1][1] = in;
    inertia[2][2] = in;
    m->inertiaMatrix = inertia;
    m->recalc();
    rigidMass->mass.endEdit();


    //-------------------- the particles attached to the rigid body
    GNode::SPtr rigidParticles = sofa::core::objectmodel::New<GNode>("rigidParticles",groot.get());

    // degrees of freedom of the skin
    MechanicalObject3::SPtr rigidParticleDOF = sofa::core::objectmodel::New<MechanicalObject3>();
    rigidParticles->addObject(rigidParticleDOF);
    rigidParticleDOF->resize(1);
    rigidParticleDOF->setName("Dof3");
    WriteAccessor< Data< VecCoord3 > > rp_x = *rigidParticleDOF->write(VecId::position() );
    rp_x[0] = Coord3(attach,0,0);

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    /*    RigidMappingRigid3_to_3* rigidMapping = new RigidMappingRigid3_to_3(rigidDOF,rigidParticleDOF);
        std::string pathobject1("@"+rigidBody->getName()+"/"+rigidDOF->getName());
        std::string pathobject2("@"+rigidParticles->getName()+"/"+rigidParticleDOF->getName());
        rigidMapping->setPathInputObject(pathobject1);
        rigidMapping->setPathOutputObject(pathobject2);
        rigidParticles->addObject( rigidMapping );
        rigidMapping->setName("Map23");*/


    // ---------------- Interaction force between the deformable and the rigid body
    StiffSpringForceField3::SPtr iff = sofa::core::objectmodel::New<StiffSpringForceField3>( DOF.get(), rigidParticleDOF.get() );
    iff->setPathObject1(deformableBody->getName()+"/"+DOF->getName());
    iff->setPathObject2(rigidParticles->getName()+"/"+rigidParticleDOF->getName());
    groot->addObject(iff);
    iff->setName("F13");
    iff->addSpring( 1,0, 100., 1., splength );




    //=========================== Init the scene
    sofa::simulation::tree::getSimulation()->init(groot.get());
    /*    groot->setAnimate(false);
        groot->setShowNormals(false);
        groot->setShowInteractionForceFields(true);
        groot->setShowMechanicalMappings(true);
        groot->setShowCollisionModels(false);
        groot->setShowBoundingCollisionModels(false);
        groot->setShowMappings(false);
        groot->setShowForceFields(true);
        groot->setShowWireFrame(false);
        groot->setShowVisualModels(true);
        groot->setShowBehaviorModels(true);*/



    //=========================== Run the main loop

    sofa::gui::GUIManager::MainLoop(groot);
}

