#include <iostream>
#include <fstream>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/component/odesolver/CGImplicitSolver.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/gui/SofaGUI.h>


#include <sofa/component/typedef/Constraint_double.h>
#include <sofa/component/typedef/Mass_double.h>
#include <sofa/component/typedef/MechanicalObject_double.h>
#include <sofa/component/typedef/Forcefield_double.h>
#include <sofa/component/typedef/Mapping.h>

typedef sofa::defaulttype::Vec3Types MyTypes;
typedef MyTypes::Deriv Vec3;
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);

    sofa::gui::SofaGUI::Init(argv[0]);

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
    MechanicalObject3d* DOF = new MechanicalObject3d;
    groot->addObject(DOF);
    DOF->resize(4);
    DOF->setName("DOF");
    MyTypes::VecCoord& x = *DOF->getX();

    x[0] = Vec3(0,10,0);
    x[1] = Vec3(10,0,0);
    x[2] = Vec3(-10*0.5,0,10*0.866);
    x[3] = Vec3(-10*0.5,0,-10*0.866);

    // Tetrahedron uniform mass
    UniformMass3d* mass = new UniformMass3d;
    groot->addObject(mass);
    mass->setMass(2);
    mass->setName("mass");

    // Tetrahedron topology
    sofa::component::topology::MeshTopology* topology = new sofa::component::topology::MeshTopology;
    groot->addObject( topology );
    topology->setName("topology");
    topology->addTetrahedron(0,1,2,3);

    // Tetrahedron constraints
    FixedConstraint3d* constraints = new FixedConstraint3d;
    groot->addObject(constraints);
    constraints->setName("constraints");
    constraints->addConstraint(0);

    // Tetrahedron force field
    TetrahedronFEMForceField3d* spring = new  TetrahedronFEMForceField3d;
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
    BarycentricMapping3d_to_Ext3f* mapping = new BarycentricMapping3d_to_Ext3f(DOF, visual);
    mapping->setName( "mapping" );
    skin->addObject(mapping);

    // Init the scene
    sofa::simulation::tree::getSimulation()->init(groot);
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
    sofa::gui::SofaGUI::MainLoop(groot);

    return 0;
}
