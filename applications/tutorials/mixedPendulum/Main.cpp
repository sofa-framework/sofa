#include <iostream>
#include <fstream>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/mass/UniformMass.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/MechanicalObject.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/component/visualmodel/OglModel.h>
#include <sofa/component/constraint/FixedConstraint.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/component/mapping/BarycentricMapping.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/forcefield/StiffSpringForceField.h>


#ifdef SOFA_GUI_FLTK
#include <sofa/gui/fltk/Main.h>
#endif
#ifdef SOFA_GUI_QT
#include <sofa/gui/qt/Main.h>
#endif

using sofa::core::Mapping;
using sofa::component::mapping::BarycentricMapping;
using sofa::component::topology::MeshTopology;
typedef sofa::component::odesolver::EulerSolver OdeSolver;
//typedef sofa::component::odesolver::StaticSolver OdeSolver;
using sofa::simulation::tree::GNode;


// deformable body
typedef sofa::defaulttype::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;
typedef sofa::component::visualmodel::GLExtVec3fTypes OglTypes;
typedef sofa::core::componentmodel::behavior::MechanicalState<MyTypes> MyMechanicalState;
typedef sofa::component::MechanicalObject<MyTypes> MyMechanicalObject;
typedef sofa::core::componentmodel::behavior::MappedModel<OglTypes> OglMappedModel;
typedef sofa::component::mass::UniformMass<MyTypes,double> MyUniformMass;
typedef BarycentricMapping< Mapping< MyMechanicalState, OglMappedModel > > MyMapping;
typedef sofa::component::constraint::FixedConstraint<MyTypes> MyFixedConstraint;
typedef sofa::component::forcefield::TetrahedronFEMForceField<MyTypes> MyTetrahedronFEMForceField;

// rigid body
typedef sofa::defaulttype::StdRigidTypes<3,double> MyRigidTypes;
typedef MyRigidTypes::Coord RigidCoord;
typedef MyRigidTypes::Quat Quaternion;
typedef sofa::defaulttype::StdRigidMass<3,double> MyRigidMass;
typedef sofa::component::mass::UniformMass<MyRigidTypes,MyRigidMass> MyRigidUniformMass;
typedef sofa::core::componentmodel::behavior::MechanicalState<MyRigidTypes> MyRigidState;
typedef sofa::component::MechanicalObject<MyRigidTypes> MyRigidMechanicalObject;
typedef sofa::core::componentmodel::behavior::MechanicalMapping<MyRigidState, MyMechanicalState >  MyRigidMapping;
typedef sofa::component::mapping::RigidMapping< MyRigidMapping >  MyRigidMechanicalMapping;

// attachment
typedef sofa::component::forcefield::StiffSpringForceField<MyTypes> MyStiffSpringForceField;


// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    parse("This is a SOFA application.")
    (argc,argv);

    // scene parameters
    double endPos = 1.;
    double attach = -1.;
    double splength = 1.;

    //-------------------- The graph root node
    GNode* groot = new sofa::simulation::tree::GNode;
    groot->setName( "root" );

    // One solver for all the graph
    OdeSolver* solver = new OdeSolver;
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // Set gravity for all the graph
    sofa::component::contextobject::Gravity* gravity =  new sofa::component::contextobject::Gravity;
    gravity->f_gravity.setValue( Vec3(0,-10,0) );
    groot->addObject(gravity);


    //-------------------- Deformable body
    GNode* deformableBody = new GNode;
    deformableBody->setName( "deformableBody" );
    groot->addChild(deformableBody);

    // degrees of freedom
    MyMechanicalObject* DOF = new MyMechanicalObject;
    deformableBody->addObject(DOF);
    DOF->resize(2);
    DOF->setName("DOF");
    MyTypes::VecCoord& x = *DOF->getX();

    x[0] = Vec3(0,0,0);
    x[1] = Vec3(endPos,0,0);

    // mass
    MyUniformMass* mass = new MyUniformMass;
    deformableBody->addObject(mass);
    mass->setMass(1);
    mass->setName("mass");

    // Fixed point
    MyFixedConstraint* constraints = new MyFixedConstraint;
    deformableBody->addObject(constraints);
    constraints->setName("constraints");
    constraints->addConstraint(0);

    // force field
    MyStiffSpringForceField* spring = new MyStiffSpringForceField;
    deformableBody->addObject(spring);
    spring->setName("internal spring");
    spring->addSpring( 1,0, 10., 1, splength );



    //-------------------- Rigid body
    GNode* rigidBody = new GNode;
    rigidBody->setName( "rigidBody" );
    groot->addChild(rigidBody);

    // degrees of freedom
    MyRigidMechanicalObject* rigidDOF = new MyRigidMechanicalObject;
    rigidBody->addObject(rigidDOF);
    rigidDOF->resize(1);
    rigidDOF->setName("rigidDOF");
    MyRigidTypes::VecCoord& rigid_x = *rigidDOF->getX();
    rigid_x[0] = RigidCoord( Vec3(endPos-attach+splength,0,0), Quaternion::identity() );

    // mass
    MyRigidUniformMass* rigidMass = new MyRigidUniformMass;
    rigidBody->addObject(rigidMass);


    //-------------------- the skin of the rigid body
    GNode* rigidSkin = new GNode;
    rigidSkin->setName( "rigidSkin" );
    rigidBody->addChild(rigidSkin);

    // degrees of freedom of the skin
    MyMechanicalObject* skinDOF = new MyMechanicalObject;
    rigidSkin->addObject(skinDOF);
    skinDOF->resize(1);
    skinDOF->setName("skinDOF");
    MyTypes::VecCoord& skin_x = *skinDOF->getX();
    skin_x[0] = Vec3(attach,0,0);

    // mapping from the rigid body DOF to the skin DOF, to rigidly attach the skin to the body
    MyRigidMechanicalMapping* rigidMapping = new MyRigidMechanicalMapping(rigidDOF,skinDOF);
    rigidSkin->addObject( rigidMapping );


    // ---------------- Attach the bodies
    MyStiffSpringForceField* iff = new MyStiffSpringForceField( DOF, skinDOF, 1000., 100. );
    groot->addObject(iff);
    iff->setName("Interaction force");
    iff->addSpring( 1,0, 10., 1., splength );



    // Init the scene
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



    //=======================================
    // Run the main loop
#ifdef SOFA_GUI_FLTK
    sofa::gui::fltk::MainLoop(argv[0],groot);
#endif
#ifdef SOFA_GUI_QT
    std::string fileName = "";
    sofa::gui::qt::MainLoop(argv[0],groot,fileName.c_str());
#endif
    return 0;
}

