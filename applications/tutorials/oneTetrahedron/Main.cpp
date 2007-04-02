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
#include <sofa/helper/system/FileRepository.h>

#ifdef SOFA_GUI_FLTK
#include <sofa/gui/fltk/Main.h>
#endif
#ifdef SOFA_GUI_QT
#include <sofa/gui/qt/Main.h>
#endif

typedef sofa::defaulttype::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;
typedef sofa::component::visualmodel::GLExtVec3fTypes OglTypes;

typedef sofa::core::componentmodel::behavior::MechanicalState<MyTypes> MyMechanicalState;
typedef sofa::core::componentmodel::behavior::MappedModel<OglTypes> OglMappedModel;

using sofa::core::Mapping;
using sofa::component::mapping::BarycentricMapping;

typedef BarycentricMapping< Mapping< MyMechanicalState, OglMappedModel > > MyMapping;

// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    parse("This is a SOFA application.")
    (argc,argv);

    // The graph root node
    sofa::simulation::tree::GNode* groot = new sofa::simulation::tree::GNode;
    groot->setName( "root" );

    // One solver for all the graph
    sofa::component::odesolver::CGImplicitSolver* solver = new sofa::component::odesolver::CGImplicitSolver;
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // Set gravity for all the graph
    sofa::component::contextobject::Gravity* gravity =  new sofa::component::contextobject::Gravity;
    gravity->f_gravity.setValue( Vec3(0,-10,0) );
    groot->addObject(gravity);

    // Tetrahedron degrees of freedom
    sofa::component::MechanicalObject<MyTypes>* DOF = new sofa::component::MechanicalObject<MyTypes>;
    groot->addObject(DOF);
    DOF->resize(4);
    DOF->setName("DOF");
    MyTypes::VecCoord& x = *DOF->getX();

    x[0] = Vec3(0,10,0);
    x[1] = Vec3(10,0,0);
    x[2] = Vec3(-10*0.5,0,10*0.866);
    x[3] = Vec3(-10*0.5,0,-10*0.866);

    // Tetrahedron uniform mass
    sofa::component::mass::UniformMass<MyTypes,double>* mass = new sofa::component::mass::UniformMass<MyTypes,double>;
    groot->addObject(mass);
    mass->setMass(2);
    mass->setName("mass");

    // Tetrahedron topology
    sofa::component::topology::MeshTopology* topology = new sofa::component::topology::MeshTopology;
    groot->addObject( topology );
    topology->setName("topology");
    topology->addTetrahedron(0,1,2,3);

    // Tetrahedron constraints
    sofa::component::constraint::FixedConstraint<MyTypes>* constraints = new sofa::component::constraint::FixedConstraint<MyTypes>;
    groot->addObject(constraints);
    constraints->setName("constraints");
    constraints->addConstraint(0);

    // Tetrahedron force field
    sofa::component::forcefield::TetrahedronFEMForceField<MyTypes>* spring = new sofa::component::forcefield::TetrahedronFEMForceField<MyTypes>;
    groot->addObject(spring);
    spring->setUpdateStiffnessMatrix(true);
    spring->setYoungModulus(1);

    // Tetrahedron skin
    sofa::simulation::tree::GNode* skin = new sofa::simulation::tree::GNode;
    skin->setName( "skin" );
    groot->addChild(skin);
    // The visual model
    sofa::component::visualmodel::OglModel* visual = new sofa::component::visualmodel::OglModel();
    visual->setName( "visual" );
    visual->load(sofa::helper::system::DataRepository.getFile("VisualModels/liver-smooth.obj"), "", "");
    visual->setColor("red");
    visual->applyScale(0.7);
    visual->applyTranslation(1.2, 0.8, 0);
    skin->addObject(visual);

    // The mapping between the tetrahedron (DOF) and the liver (visual)
    MyMapping* mapping = new MyMapping(DOF, visual);
    mapping->setName( "mapping" );
    skin->addObject(mapping);

    // Init the scene
    sofa::simulation::tree::Simulation::init(groot);
    groot->setAnimate(false);
    groot->setShowNormals(false);
    groot->setShowInteractionForceFields(false);
    groot->setShowMechanicalMappings(false);
    groot->setShowCollisionModels(false);
    groot->setShowBoundingCollisionModels(false);
    groot->setShowMappings(false);
    groot->setShowForceFields(true);
    groot->setShowWireFrame(true);
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
