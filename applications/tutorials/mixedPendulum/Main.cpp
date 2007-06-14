/** A sample program. Laure Heigeas, Francois Faure, 2007. */
// scene data structure
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/constraint/FixedConstraint.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/mapping/RigidMapping.h>
// gui
#include <sofa/gui/SofaGUI.h>

using sofa::simulation::tree::GNode;
typedef sofa::component::odesolver::EulerSolver OdeSolver;
using sofa::component::contextobject::Gravity;

// deformable body
typedef sofa::defaulttype::Vec3Types ParticleTypes;
typedef ParticleTypes::Deriv Vec3;
typedef sofa::core::componentmodel::behavior::MechanicalState<ParticleTypes> ParticleStates;
typedef sofa::component::MechanicalObject<ParticleTypes> ParticleDOFs;
typedef sofa::component::mass::UniformMass<ParticleTypes,double> ParticleMasses;
typedef sofa::component::constraint::FixedConstraint<ParticleTypes> ParticleFixedConstraint;
typedef sofa::component::forcefield::StiffSpringForceField<ParticleTypes> ParticleStiffSpringForceField;
typedef sofa::component::visualmodel::GLExtVec3fTypes OglTypes;
typedef sofa::core::componentmodel::behavior::MappedModel<OglTypes> OglMappedModel;

// rigid body
typedef sofa::defaulttype::StdRigidTypes<3,double> RigidTypes;
typedef RigidTypes::Coord RigidCoord;
typedef RigidTypes::Quat Quaternion;
typedef sofa::defaulttype::StdRigidMass<3,double> RigidMass;
typedef sofa::component::mass::UniformMass<RigidTypes,RigidMass> RigidUniformMasses;
typedef sofa::core::componentmodel::behavior::MechanicalState<RigidTypes> RigidStates;
typedef sofa::component::MechanicalObject<RigidTypes> RigidDOFs;
typedef sofa::core::componentmodel::behavior::MechanicalMapping<RigidStates, ParticleStates >  RigidToParticleMechanicalMapping;
typedef sofa::component::mapping::RigidMapping< RigidToParticleMechanicalMapping >  RigidToParticleRigidMechanicalMapping;

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
    solver->f_printLog.setValue(false);

    // Set gravity for all the graph
    Gravity* gravity =  new Gravity;
    groot->addObject(gravity);
    gravity->f_gravity.setValue( Vec3(0,-10,0) );


    //-------------------- Deformable body
    GNode* deformableBody = new GNode;
    groot->addChild(deformableBody);
    deformableBody->setName( "deformableBody" );

    // degrees of freedom
    ParticleDOFs* DOF = new ParticleDOFs;
    deformableBody->addObject(DOF);
    DOF->resize(2);
    DOF->setName("DOF");
    ParticleTypes::VecCoord& x = *DOF->getX();
    x[0] = Vec3(0,0,0);
    x[1] = Vec3(endPos,0,0);

    // mass
    ParticleMasses* mass = new ParticleMasses;
    deformableBody->addObject(mass);
    mass->setMass(1);
    mass->setName("mass");

    // Fixed point
    ParticleFixedConstraint* constraints = new ParticleFixedConstraint;
    deformableBody->addObject(constraints);
    constraints->setName("constraints");
    constraints->addConstraint(0);

    // force field
    ParticleStiffSpringForceField* spring = new ParticleStiffSpringForceField;
    deformableBody->addObject(spring);
    spring->setName("internal spring");
    spring->addSpring( 1,0, 10., 1, splength );


    //-------------------- Rigid body
    GNode* rigidBody = new GNode;
    groot->addChild(rigidBody);
    rigidBody->setName( "rigidBody" );

    // degrees of freedom
    RigidDOFs* rigidDOF = new RigidDOFs;
    rigidBody->addObject(rigidDOF);
    rigidDOF->resize(1);
    rigidDOF->setName("rigidDOF");
    RigidTypes::VecCoord& rigid_x = *rigidDOF->getX();
    rigid_x[0] = RigidCoord( Vec3(endPos-attach+splength,0,0), Quaternion::identity() );

    // mass
    RigidUniformMasses* rigidMass = new RigidUniformMasses;
    rigidBody->addObject(rigidMass);


    //-------------------- the particles attached to the rigid body
    GNode* rigidParticles = new GNode;
    rigidParticles->setName( "rigidParticles" );
    rigidBody->addChild(rigidParticles);

    // degrees of freedom of the skin
    ParticleDOFs* rigidParticleDOF = new ParticleDOFs;
    rigidParticles->addObject(rigidParticleDOF);
    rigidParticleDOF->resize(1);
    rigidParticleDOF->setName("rigidParticleDOF");
    ParticleTypes::VecCoord& rp_x = *rigidParticleDOF->getX();
    rp_x[0] = Vec3(attach,0,0);

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    RigidToParticleRigidMechanicalMapping* rigidMapping = new RigidToParticleRigidMechanicalMapping(rigidDOF,rigidParticleDOF);
    rigidParticles->addObject( rigidMapping );


    // ---------------- Interaction force between the deformable and the rigid body
    ParticleStiffSpringForceField* iff = new ParticleStiffSpringForceField( DOF, rigidParticleDOF );
    groot->addObject(iff);
    iff->setName("Interaction force");
    iff->addSpring( 1,0, 10., 1., splength );



    //=========================== Init the scene
    sofa::simulation::tree::Simulation::init(groot);
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

