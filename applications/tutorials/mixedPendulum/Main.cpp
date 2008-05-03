/** A sample program. Laure Heigeas, Francois Faure, 2007. */
// scene data structure
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/visualmodel/OglModel.h>
// gui
#include <sofa/gui/SofaGUI.h>

#include <sofa/component/typedef/Sofa_typedef.h>

using namespace sofa::simulation::tree;
using sofa::component::odesolver::EulerSolver;

int main(int, char** argv)
{
    sofa::gui::SofaGUI::Init(argv[0]);
    //=========================== Build the scene
    double endPos = 1.;
    double attach = -1.;
    double splength = 1.;

    //-------------------- The graph root node
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Coord3(0,-10,0) );

    // One solver for all the graph
    EulerSolver* solver = new EulerSolver;
    groot->addObject(solver);
    solver->setName("S");

    //-------------------- Deformable body
    GNode* deformableBody = new GNode;
    groot->addChild(deformableBody);
    deformableBody->setName( "deformableBody" );

    // degrees of freedom
    MechanicalObject3* DOF = new MechanicalObject3;
    deformableBody->addObject(DOF);
    DOF->resize(2);
    DOF->setName("Dof1");
    VecCoord3& x = *DOF->getX();
    x[0] = Coord3(0,0,0);
    x[1] = Coord3(endPos,0,0);

    // mass
    //    ParticleMasses* mass = new ParticleMasses;
    UniformMass3* mass = new UniformMass3;
    deformableBody->addObject(mass);
    mass->setMass(1);
    mass->setName("M1");

    // Fixed point
    FixedConstraint3* constraints = new FixedConstraint3;
    deformableBody->addObject(constraints);
    constraints->setName("C");
    constraints->addConstraint(0);


    // force field
    StiffSpringForceField3* spring = new StiffSpringForceField3;
    deformableBody->addObject(spring);
    spring->setName("F1");
    spring->addSpring( 1,0, 10., 1, splength );


    //-------------------- Rigid body
    GNode* rigidBody = new GNode;
    groot->addChild(rigidBody);
    rigidBody->setName( "rigidBody" );

    // degrees of freedom
    MechanicalObjectRigid3* rigidDOF = new MechanicalObjectRigid3;
    rigidBody->addObject(rigidDOF);
    rigidDOF->resize(1);
    rigidDOF->setName("Dof2");
    VecCoordRigid3& rigid_x = *rigidDOF->getX();
    rigid_x[0] = CoordRigid3( Coord3(endPos-attach+splength,0,0),
            Quat3::identity() );

    // mass
    UniformMassRigid3* rigidMass = new UniformMassRigid3;
    rigidBody->addObject(rigidMass);
    rigidMass->setName("M2");


    //-------------------- the particles attached to the rigid body
    GNode* rigidParticles = new GNode;
    rigidParticles->setName( "rigidParticles" );
    rigidBody->addChild(rigidParticles);

    // degrees of freedom of the skin
    MechanicalObject3* rigidParticleDOF = new MechanicalObject3;
    rigidParticles->addObject(rigidParticleDOF);
    rigidParticleDOF->resize(1);
    rigidParticleDOF->setName("Dof3");
    VecCoord3& rp_x = *rigidParticleDOF->getX();
    rp_x[0] = Coord3(attach,0,0);

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    RigidMechanicalMappingRigid3_to_3* rigidMapping = new RigidMechanicalMappingRigid3_to_3(rigidDOF,rigidParticleDOF);
    rigidParticles->addObject( rigidMapping );
    rigidMapping->setName("Map23");


    // ---------------- Interaction force between the deformable and the rigid body
    StiffSpringForceField3* iff = new StiffSpringForceField3( DOF, rigidParticleDOF );
    groot->addObject(iff);
    iff->setName("F13");
    iff->addSpring( 1,0, 10., 1., splength );




    //=========================== Init the scene
    getSimulation()->init(groot);
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
    groot->setShowBehaviorModels(true);



    //=========================== Run the main loop

    sofa::gui::SofaGUI::MainLoop(groot);
}

