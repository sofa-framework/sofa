#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/Simulation.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/gui/SofaGUI.h>

//Including components for collision detection
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/TriangleModel.h>

//Including component for topological description of the objects
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>

//Including Solvers
#include <sofa/component/odesolver/CGImplicitSolver.h>


#include <sofa/component/visualmodel/OglModel.h>



using sofa::component::visualmodel::OglModel;

using namespace sofa::simulation::tree;
using namespace sofa::component::collision;
using namespace sofa::component::topology;
using sofa::component::odesolver::CGImplicitSolver;

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------
int main(int argc, char** argv)
{
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);
    sofa::gui::SofaGUI::Init(argv[0]);

    // The graph root node
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Coord3(0,-10,0) );


    //Components for collision management
    //------------------------------------
    //--> adding collision pipeline
    DefaultPipeline* collisionPipeline = new DefaultPipeline;
    collisionPipeline->setName("Collision Pipeline");
    groot->addObject(collisionPipeline);

    //--> adding collision detection system
    BruteForceDetection* detection = new BruteForceDetection;
    detection->setName("Detection");
    groot->addObject(detection);

    //--> adding component to detection intersection of elements
    NewProximityIntersection* detectionProximity = new NewProximityIntersection;
    detectionProximity->setName("Proximity");
    detectionProximity->setAlarmDistance(0.3);   //warning distance
    detectionProximity->setContactDistance(0.2); //min distance before setting a spring to create a repulsion
    groot->addObject(detectionProximity);


    //--> adding contact manager
    DefaultContactManager* contactManager = new DefaultContactManager;
    contactManager->setName("Contact Manager");
    groot->addObject(contactManager);

    //--> adding component to handle groups of collision.
    DefaultCollisionGroupManager* collisionGroupManager = new DefaultCollisionGroupManager;
    collisionGroupManager->setName("Collision Group Manager");
    groot->addObject(collisionGroupManager);



    //Elements of the scene
    //------------------------------------
    GNode* chain = new GNode;
    chain->setName("Chain");
    groot->addChild(chain);

    //************************************
    //Torus Fixed
    GNode* torusFixed = new GNode;
    torusFixed->setName("Torus Fixed");
    chain->addChild(torusFixed);

    MeshTopology* meshTorusFixed = new MeshTopology;
    meshTorusFixed->load(sofa::helper::system::DataRepository.getFile("CollisionModels/torus2_for_collision.obj").c_str());
    torusFixed->addObject(meshTorusFixed);

    MechanicalObject3* dofFixed = new MechanicalObject3;
    torusFixed->addObject(dofFixed);

    TriangleModel* triangleFixed = new TriangleModel;
    triangleFixed->setSimulated(false); //Not simulated, fixed object
    triangleFixed->setMoving(false);    //No extern events
    torusFixed->addObject(triangleFixed);

    OglModel* visualFixed = new OglModel;
    visualFixed->setName("visual");
    visualFixed->load(sofa::helper::system::DataRepository.getFile("VisualModels/torus2.obj"),"","");
    visualFixed->setColor("gray");
    torusFixed->addObject(visualFixed);


    //************************************
    //Torus FEM
    GNode* torusFEM = new GNode;
    torusFEM->setName("Torus FEM");
    chain->addChild(torusFEM);

    CGImplicitSolver* solverFEM = new CGImplicitSolver;
    solverFEM->setName("Conjugate Gradient Implicit");
    solverFEM->f_maxIter.setValue(20); //iteration maxi for the CG
    solverFEM->f_smallDenominatorThreshold.setValue(0.000001);
    solverFEM->f_tolerance.setValue(0.001);
    solverFEM->f_rayleighStiffness.setValue(0.01);
    solverFEM->f_rayleighMass.setValue(1);
    torusFEM->addObject(solverFEM);

    MeshTopology* meshTorusFEM = new MeshTopology;
    meshTorusFEM->load(sofa::helper::system::DataRepository.getFile("Topology/torus_low_res.msh").c_str());
    torusFEM->addObject(meshTorusFEM);

    MechanicalObject3* dofFEM = new MechanicalObject3;
    dofFEM->applyTranslation(2.5,0,0);
    torusFEM->addObject(dofFEM);

    UniformMass3* uniMassFEM = new UniformMass3;
    uniMassFEM->setTotalMass(5); //the whole object will have 5 as given mass
    torusFEM->addObject(uniMassFEM);

    TetrahedronFEMForceField3* tetraFEMFF = new TetrahedronFEMForceField3;
    tetraFEMFF->setName("FEM");
    tetraFEMFF->setComputeGlobalMatrix(false);
    tetraFEMFF->setMethod("large");
    tetraFEMFF->setPoissonRatio(0.3);
    tetraFEMFF->setYoungModulus(1000);
    torusFEM->addObject(tetraFEMFF);

    //Node VISUAL
    GNode* FEMVisualNode = new GNode;
    FEMVisualNode->setName("Visu");
    torusFEM->addChild(FEMVisualNode);

    OglModel* visualFEM = new OglModel;
    visualFEM->setName("visual");
    visualFEM->load(sofa::helper::system::DataRepository.getFile("VisualModels/torus.obj"),"","");
    visualFEM->setColor("red");
    visualFEM->applyTranslation(2.5,0,0);
    FEMVisualNode->addObject(visualFEM);

    BarycentricMapping3_to_Ext3* mappingFEM = new BarycentricMapping3_to_Ext3(dofFEM, visualFEM);
    mappingFEM->setName("Mapping Visual");
    FEMVisualNode->addObject(mappingFEM);


    //Node COLLISION
    GNode* FEMCollisionNode = new GNode;
    FEMCollisionNode->setName("Visu");
    torusFEM->addChild(FEMCollisionNode);

    MeshTopology* meshTorusFEM_surf= new MeshTopology;
    meshTorusFEM_surf->load(sofa::helper::system::DataRepository.getFile("CollisionModels/torus_for_collision.obj").c_str());
    FEMCollisionNode->addObject(meshTorusFEM_surf);

    MechanicalObject3* dofFEM_surf = new MechanicalObject3;
    dofFEM_surf->applyTranslation(2.5,0,0);
    FEMCollisionNode->addObject(dofFEM_surf);

    TriangleModel* triangleFEM = new TriangleModel;
    FEMCollisionNode->addObject(triangleFEM);

    BarycentricMechanicalMapping3_to_3* mechaMappingFEM = new BarycentricMechanicalMapping3_to_3(dofFEM, dofFEM_surf);
    FEMCollisionNode->addObject(mechaMappingFEM);


    //************************************
    //Torus Spring
    GNode* torusSpring = new GNode;
    torusSpring->setName("Torus Spring");
    chain->addChild(torusSpring);

    CGImplicitSolver* solverSpring = new CGImplicitSolver;
    solverSpring->setName("Conjugate Gradient Implicit");
    solverSpring->f_maxIter.setValue(20); //iteration maxi for the CG
    solverSpring->f_smallDenominatorThreshold.setValue(0.000001);
    solverSpring->f_tolerance.setValue(0.001);
    solverSpring->f_rayleighStiffness.setValue(0.01);
    solverSpring->f_rayleighMass.setValue(1);
    torusSpring->addObject(solverSpring);

    MeshTopology* meshTorusSpring = new MeshTopology;
    meshTorusSpring->load(sofa::helper::system::DataRepository.getFile("Topology/torus2_low_res.msh").c_str());
    torusSpring->addObject(meshTorusSpring);

    MechanicalObject3* dofSpring = new MechanicalObject3;
    dofSpring->applyTranslation(5,0,0);
    torusSpring->addObject(dofSpring);

    UniformMass3* uniMassSpring = new UniformMass3;
    uniMassSpring->setTotalMass(5); //the whole object will have 5 as given mass
    torusSpring->addObject(uniMassSpring);

    MeshSpringForceField3* springFF = new MeshSpringForceField3;
    springFF->setName("Springs");
    springFF->setStiffness(400);
    springFF->setDamping(0);
    torusSpring->addObject(springFF);

    //Node VISUAL
    GNode* SpringVisualNode = new GNode;
    SpringVisualNode->setName("Visu");
    torusSpring->addChild(SpringVisualNode);

    OglModel* visualSpring = new OglModel;
    visualSpring->setName("visual");
    visualSpring->load(sofa::helper::system::DataRepository.getFile("VisualModels/torus2.obj"),"","");
    visualSpring->setColor("green");
    visualSpring->applyTranslation(5,0,0);
    SpringVisualNode->addObject(visualSpring);

    BarycentricMapping3_to_Ext3* mappingSpring = new BarycentricMapping3_to_Ext3(dofSpring, visualSpring);
    mappingSpring->setName("Mapping Visual");
    SpringVisualNode->addObject(mappingSpring);


    //Node COLLISION
    GNode* SpringCollisionNode = new GNode;
    SpringCollisionNode->setName("Visu");
    torusSpring->addChild(SpringCollisionNode);

    MeshTopology* meshTorusSpring_surf= new MeshTopology;
    meshTorusSpring_surf->load(sofa::helper::system::DataRepository.getFile("CollisionModels/torus2_for_collision.obj").c_str());
    SpringCollisionNode->addObject(meshTorusSpring_surf);

    MechanicalObject3* dofSpring_surf = new MechanicalObject3;
    dofSpring_surf->applyTranslation(5,0,0);
    SpringCollisionNode->addObject(dofSpring_surf);

    TriangleModel* triangleSpring = new TriangleModel;
    SpringCollisionNode->addObject(triangleSpring);

    BarycentricMechanicalMapping3_to_3* mechaMappingSpring = new BarycentricMechanicalMapping3_to_3(dofSpring, dofSpring_surf);
    SpringCollisionNode->addObject(mechaMappingSpring);

    //************************************
    //Torus FFD
    GNode* torusFFD = new GNode;
    torusFFD->setName("Torus FFD");
    chain->addChild(torusFFD);

    CGImplicitSolver* solverFFD = new CGImplicitSolver;
    solverFFD->setName("Conjugate Gradient Implicit");
    solverFFD->f_maxIter.setValue(20); //iteration maxi for the CG
    solverFFD->f_smallDenominatorThreshold.setValue(0.000001);
    solverFFD->f_tolerance.setValue(0.001);
    solverFFD->f_rayleighStiffness.setValue(0.01);
    solverFFD->f_rayleighMass.setValue(1);
    torusFFD->addObject(solverFFD);

    MechanicalObject3* dofFFD = new MechanicalObject3;
    dofFFD->applyTranslation(7.5,0,0);
    torusFFD->addObject(dofFFD);

    UniformMass3* uniMassFFD = new UniformMass3;
    uniMassFFD->setTotalMass(5); //the whole object will have 5 as given mass
    torusFFD->addObject(uniMassFFD);

    RegularGridTopology* gridTopo = new RegularGridTopology(6,2,5); //dimension of the grid
    gridTopo->setPos(
        -2.5,2.5,  //Xmin, Xmax
        -0.5,0.5,  //Ymin, Ymax
        -2,2       //Zmin, Zmax
    );
    torusFFD->addObject(gridTopo);

    RegularGridSpringForceField3* FFDFF = new RegularGridSpringForceField3;
    FFDFF->setName("Springs FFD");
    FFDFF->setStiffness(200);
    FFDFF->setDamping(0);
    torusFFD->addObject(FFDFF);

    //Node VISUAL
    GNode* FFDVisualNode = new GNode;
    FFDVisualNode->setName("Visu");
    torusFFD->addChild(FFDVisualNode);

    OglModel* visualFFD = new OglModel;
    visualFFD->setName("visual");
    visualFFD->load(sofa::helper::system::DataRepository.getFile("VisualModels/torus.obj"),"","");
    visualFFD->setColor("yellow");
    FFDVisualNode->addObject(visualFFD);

    BarycentricMapping3_to_Ext3* mappingFFD = new BarycentricMapping3_to_Ext3(dofFFD, visualFFD);
    mappingFFD->setName("Mapping Visual");
    FFDVisualNode->addObject(mappingFFD);


    //Node COLLISION
    GNode* FFDCollisionNode = new GNode;
    FFDCollisionNode->setName("Visu");
    torusFFD->addChild(FFDCollisionNode);

    MeshTopology* meshTorusFFD_surf= new MeshTopology;
    meshTorusFFD_surf->load(sofa::helper::system::DataRepository.getFile("CollisionModels/torus_for_collision.obj").c_str());
    FFDCollisionNode->addObject(meshTorusFFD_surf);

    MechanicalObject3* dofFFD_surf = new MechanicalObject3;
    FFDCollisionNode->addObject(dofFFD_surf);

    TriangleModel* triangleFFD = new TriangleModel;
    FFDCollisionNode->addObject(triangleFFD);

    BarycentricMechanicalMapping3_to_3* mechaMappingFFD = new BarycentricMechanicalMapping3_to_3(dofFFD, dofFFD_surf);
    FFDCollisionNode->addObject(mechaMappingFFD);




    //************************************
    //Torus Rigid
    GNode* torusRigid = new GNode;
    torusRigid->setName("Torus Rigid");
    chain->addChild(torusRigid);

    CGImplicitSolver* solverRigid = new CGImplicitSolver;
    solverRigid->setName("Conjugate Gradient Implicit");
    solverRigid->f_maxIter.setValue(20); //iteration maxi for the CG
    solverRigid->f_smallDenominatorThreshold.setValue(0.000001);
    solverRigid->f_tolerance.setValue(0.001);
    solverRigid->f_rayleighStiffness.setValue(0.01);
    solverRigid->f_rayleighMass.setValue(1);
    torusRigid->addObject(solverRigid);

    MechanicalObjectRigid3* dofRigid = new MechanicalObjectRigid3;
    dofRigid->applyTranslation(10,0,0);
    torusRigid->addObject(dofRigid);

    UniformMassRigid3* uniMassRigid = new UniformMassRigid3;
    uniMassRigid->setTotalMass(1); //the whole object will have 5 as given mass
    torusRigid->addObject(uniMassRigid);

    //Node VISUAL
    GNode* RigidVisualNode = new GNode;
    RigidVisualNode->setName("Visu");
    torusRigid->addChild(RigidVisualNode);

    OglModel* visualRigid = new OglModel;
    visualRigid->setName("visual");
    visualRigid->load(sofa::helper::system::DataRepository.getFile("VisualModels/torus2.obj"),"","");
    visualRigid->setColor("gray");
    RigidVisualNode->addObject(visualRigid);

    RigidMappingRigid3_to_Ext3* mappingRigid = new RigidMappingRigid3_to_Ext3(dofRigid, visualRigid);
    mappingRigid->setName("Mapping Visual");
    RigidVisualNode->addObject(mappingRigid);


    //Node COLLISION
    GNode* RigidCollisionNode = new GNode;
    RigidCollisionNode->setName("Visu");
    torusRigid->addChild(RigidCollisionNode);

    MeshTopology* meshTorusRigid_surf= new MeshTopology;
    meshTorusRigid_surf->load(sofa::helper::system::DataRepository.getFile("CollisionModels/torus2_for_collision.obj").c_str());
    RigidCollisionNode->addObject(meshTorusRigid_surf);

    MechanicalObject3* dofRigid_surf = new MechanicalObject3;
    RigidCollisionNode->addObject(dofRigid_surf);

    TriangleModel* triangleRigid = new TriangleModel;
    RigidCollisionNode->addObject(triangleRigid);

    RigidMechanicalMappingRigid3_to_3* mechaMappingRigid = new RigidMechanicalMappingRigid3_to_3(dofRigid, dofRigid_surf);
    RigidCollisionNode->addObject(mechaMappingRigid);



    getSimulation()->init(groot);
    groot->setAnimate(false);

    //=======================================
    // Run the main loop
    sofa::gui::SofaGUI::MainLoop(groot);

    return 0;
}
