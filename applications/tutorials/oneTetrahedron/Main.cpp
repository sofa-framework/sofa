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

#include <sofa/component/typedef/Sofa_typedef.h>

using namespace sofa::simulation::tree;
using sofa::component::odesolver::CGImplicitSolver;
using sofa::component::topology::MeshTopology;
using sofa::component::visualmodel::OglModel;
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);

    sofa::gui::SofaGUI::Init(argv[0]);

    // The graph root node : gravity already exists in a GNode by default
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Coord3(0,-10,0) );


    // One solver for all the graph
    CGImplicitSolver* solver = new CGImplicitSolver;
    solver->f_printLog.setValue(false);
    groot->addObject(solver);

    // Tetrahedron degrees of freedom
    MechanicalObject3* DOF = new MechanicalObject3;
    groot->addObject(DOF);
    DOF->resize(4);
    DOF->setName("DOF");
    VecCoord3& x = *DOF->getX();

    x[0] = Coord3(0,10,0);
    x[1] = Coord3(10,0,0);
    x[2] = Coord3(-10*0.5,0,10*0.866);
    x[3] = Coord3(-10*0.5,0,-10*0.866);

    // Tetrahedron uniform mass
    UniformMass3* mass = new UniformMass3;
    groot->addObject(mass);
    mass->setMass(2);
    mass->setName("mass");

    // Tetrahedron topology
    MeshTopology* topology = new MeshTopology;
    groot->addObject( topology );
    topology->setName("topology");
    topology->addTetrahedron(0,1,2,3);

    // Tetrahedron constraints
    FixedConstraint3* constraints = new FixedConstraint3;
    groot->addObject(constraints);
    constraints->setName("constraints");
    constraints->addConstraint(0);

    // Tetrahedron force field
    TetrahedronFEMForceField3* spring = new  TetrahedronFEMForceField3;
    groot->addObject(spring);
    spring->setUpdateStiffnessMatrix(true);
    spring->setYoungModulus(6);

    // Tetrahedron skin
    GNode* skin = new GNode;
    skin->setName( "skin" );
    groot->addChild(skin);
    // The visual model
    OglModel* visual = new OglModel();
    visual->setName( "visual" );
    visual->load(sofa::helper::system::DataRepository.getFile("VisualModels/liver-smooth.obj"), "", "");
    visual->setColor("red");
    visual->applyScale(0.7);
    visual->applyTranslation(1.2, 0.8, 0);
    skin->addObject(visual);

    // The mapping between the tetrahedron (DOF) and the liver (visual)
    BarycentricMapping3_to_Ext3* mapping = new BarycentricMapping3_to_Ext3(DOF, visual);
    mapping->setName( "mapping" );
    skin->addObject(mapping);

    // Init the scene
    getSimulation()->init(groot);
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
