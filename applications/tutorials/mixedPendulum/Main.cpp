/** A sample program. Laure Heigeas, Francois Faure, 2007. */
// scene data structure
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>

#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/mapping/RigidMapping.h>
// gui
#include <sofa/gui/SofaGUI.h>

#include <sofa/component/typedef/MechanicalObject_double.h>
#include <sofa/component/typedef/Mass_double.h>
#include <sofa/component/typedef/Constraint_double.h>
#include <sofa/component/typedef/Forcefield_double.h>
#include <sofa/component/typedef/Mapping.h>

using sofa::simulation::tree::GNode;
typedef sofa::component::odesolver::EulerSolver OdeSolver;
using sofa::component::contextobject::Gravity;

// deformable body
typedef sofa::defaulttype::Vec3Types ParticleTypes;
typedef ParticleTypes::Deriv Vec3;
// rigid body
typedef sofa::defaulttype::StdRigidTypes<3,double>::Coord RigidCoord;
typedef sofa::defaulttype::StdRigidTypes<3,double>::Quat Quaternion;

int main(int, char** argv)
{
    sofa::gui::SofaGUI::Init(argv[0]);
    //=========================== Build the scene
    double endPos = 1.;
    double attach = -1.;
    double splength = 1.;

    //-------------------- The graph root node
    GNode* groot = new sofa::simulation::tree::GNode;
    groot->setName( "root" );

    // One solver for all the graph
    OdeSolver* solver = new OdeSolver;
    groot->addObject(solver);
    solver->setName("S");

    //-------------------- Deformable body
    GNode* deformableBody = new GNode;
    groot->addChild(deformableBody);
    deformableBody->setName( "deformableBody" );

    // degrees of freedom
    MechanicalObject3d* DOF = new MechanicalObject3d;
    deformableBody->addObject(DOF);
    DOF->resize(2);
    DOF->setName("Dof1");
    ParticleTypes::VecCoord& x = *DOF->getX();
    x[0] = Vec3(0,0,0);
    x[1] = Vec3(endPos,0,0);

    // mass
    //    ParticleMasses* mass = new ParticleMasses;
    UniformMass3d* mass = new UniformMass3d;
    deformableBody->addObject(mass);
    mass->setMass(1);
    mass->setName("M1");

    // Fixed point
    FixedConstraint3d* constraints = new FixedConstraint3d;
    deformableBody->addObject(constraints);
    constraints->setName("C");
    constraints->addConstraint(0);

    // force field
    StiffSpringForceField3d* spring = new StiffSpringForceField3d;
    deformableBody->addObject(spring);
    spring->setName("F1");
    spring->addSpring( 1,0, 10., 1, splength );


    //-------------------- Rigid body
    GNode* rigidBody = new GNode;
    groot->addChild(rigidBody);
    rigidBody->setName( "rigidBody" );

    // degrees of freedom
    MechanicalObjectRigid3d* rigidDOF = new MechanicalObjectRigid3d;
    rigidBody->addObject(rigidDOF);
    rigidDOF->resize(1);
    rigidDOF->setName("Dof2");
    sofa::defaulttype::StdRigidTypes<3,double>::VecCoord& rigid_x = *rigidDOF->getX();
    rigid_x[0] = sofa::defaulttype::StdRigidTypes<3,double>::Coord( Vec3(endPos-attach+splength,0,0),
            sofa::defaulttype::StdRigidTypes<3,double>::Quat::identity() );

    // mass
    UniformMassRigid3d* rigidMass = new UniformMassRigid3d;
    rigidBody->addObject(rigidMass);
    rigidMass->setName("M2");


    //-------------------- the particles attached to the rigid body
    GNode* rigidParticles = new GNode;
    rigidParticles->setName( "rigidParticles" );
    rigidBody->addChild(rigidParticles);

    // degrees of freedom of the skin
    MechanicalObject3d* rigidParticleDOF = new MechanicalObject3d;
    rigidParticles->addObject(rigidParticleDOF);
    rigidParticleDOF->resize(1);
    rigidParticleDOF->setName("Dof3");
    ParticleTypes::VecCoord& rp_x = *rigidParticleDOF->getX();
    rp_x[0] = Vec3(attach,0,0);

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    RigidMechanicalMappingRigid3d_to_3d* rigidMapping = new RigidMechanicalMappingRigid3d_to_3d(rigidDOF,rigidParticleDOF);
    rigidParticles->addObject( rigidMapping );
    rigidMapping->setName("Map23");


    // ---------------- Interaction force between the deformable and the rigid body
    StiffSpringForceField3d* iff = new StiffSpringForceField3d( DOF, rigidParticleDOF );
    groot->addObject(iff);
    iff->setName("F13");
    iff->addSpring( 1,0, 10., 1., splength );

    // Set gravity for the whole graph
    Gravity* gravity =  new Gravity;
    groot->addObject(gravity);
    gravity->f_gravity.setValue( Vec3(0,-10,0) );



    //=========================== Init the scene
    sofa::simulation::tree::getSimulation()->init(groot);
    groot->setAnimate(false);
    groot->setShowNormals(false);
    groot->setShowInteractionForceFields(true);
    groot->setShowMechanicalMappings(true);
    groot->setShowCollisionModels(false);
    groot->setShowBoundingCollisionModels(false);
    groot->setShowMappings(false);
    groot->setShowForceFields(true);
    groot->setShowWireFrame(false);
    groot->setShowVisualModels(true);



    //=========================== Run the main loop

    sofa::gui::SofaGUI::MainLoop(groot);
}

