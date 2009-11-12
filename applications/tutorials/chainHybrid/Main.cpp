/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/


#include <sofa/helper/ArgumentParser.h>
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/TreeSimulation.h>



#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/gui/GUIManager.h>

//Including components for collision detection
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/TreeCollisionGroupManager.h>
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/TriangleModel.h>

//Including component for topological description of the objects
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/container/MeshLoader.h>

//Including Solvers
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>

#include <sofa/component/visualmodel/OglModel.h>

#include <sofa/simulation/common/TransformationVisitor.h>
#include <sofa/helper/system/glut.h>

#include <sofa/helper/system/SetDirectory.h>


//SOFA_HAS_BOOST_KERNEL to define in chainHybrid.pro

#ifdef SOFA_HAS_BOOST_KERNEL
#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/component/collision/BglCollisionGroupManager.h>
#else
#include <sofa/simulation/tree/GNode.h>
#endif


using sofa::component::visualmodel::OglModel;

using namespace sofa::simulation;
using namespace sofa::component::collision;
using namespace sofa::component::topology;
using sofa::component::container::MeshLoader;
using sofa::component::odesolver::EulerImplicitSolver;
using sofa::component::linearsolver::CGLinearSolver;
using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;
typedef CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> CGLinearSolverGraph;


//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------


Node *createChainHybrid(std::string
#ifdef SOFA_HAS_BOOST_KERNEL
        simulationType
#endif
                       )
{
    // The graph root node
    Node* root = getSimulation()->newNode("root");
    root->setGravityInWorld( Coord3(0,0,-10) );


    //Components for collision management
    //------------------------------------
    //--> adding collision pipeline
    DefaultPipeline* collisionPipeline = new DefaultPipeline;
    collisionPipeline->setName("Collision Pipeline");
    root->addObject(collisionPipeline);

    //--> adding collision detection system
    BruteForceDetection* detection = new BruteForceDetection;
    detection->setName("Detection");
    root->addObject(detection);

    //--> adding component to detection intersection of elements
    NewProximityIntersection* detectionProximity = new NewProximityIntersection;
    detectionProximity->setName("Proximity");
    detectionProximity->setAlarmDistance(0.3);   //warning distance
    detectionProximity->setContactDistance(0.2); //min distance before setting a spring to create a repulsion
    root->addObject(detectionProximity);

    //--> adding contact manager
    DefaultContactManager* contactManager = new DefaultContactManager;
    contactManager->setName("Contact Manager");
    root->addObject(contactManager);

    //--> adding component to handle groups of collision.
#ifdef SOFA_HAS_BOOST_KERNEL
    if (simulationType == "bgl")
    {
        BglCollisionGroupManager* collisionGroupManager = new BglCollisionGroupManager;
        collisionGroupManager->setName("Collision Group Manager");
        root->addObject(collisionGroupManager);
    }
    else
#endif
    {
        //--> adding component to handle groups of collision.
        TreeCollisionGroupManager* collisionGroupManager = new TreeCollisionGroupManager;
        collisionGroupManager->setName("Collision Group Manager");
        root->addObject(collisionGroupManager);
    }

    //Elements of the scene
    //------------------------------------
    Node* chain = getSimulation()->newNode("Chain");
    root->addChild(chain);

    //************************************
    //Torus Fixed
    Node* torusFixed = getSimulation()->newNode("Fixed");
    chain->addChild(torusFixed);

    MeshLoader* loaderFixed = new MeshLoader;
    loaderFixed->load(sofa::helper::system::DataRepository.getFile("mesh/torus_for_collision.obj").c_str());
    torusFixed->addObject(loaderFixed);

    MeshTopology* meshTorusFixed = new MeshTopology;
    torusFixed->addObject(meshTorusFixed);

    MechanicalObject3* dofFixed = new MechanicalObject3; dofFixed->setName("Fixed Object");
    torusFixed->addObject(dofFixed);

    TriangleModel* triangleFixed = new TriangleModel; triangleFixed->setName("Collision Fixed");
    triangleFixed->setSimulated(false); //Not simulated, fixed object
    triangleFixed->setMoving(false);    //No extern events
    torusFixed->addObject(triangleFixed);

    OglModel* visualFixed = new OglModel;
    visualFixed->setName("visual");
    visualFixed->load(sofa::helper::system::DataRepository.getFile("mesh/torus.obj"),"","");
    visualFixed->setColor("gray");
    torusFixed->addObject(visualFixed);

    //************************************
    //Torus FEM
    Node* torusFEM = getSimulation()->newNode("FEM");
    chain->addChild(torusFEM);


    EulerImplicitSolver* solverFEM = new EulerImplicitSolver;
    CGLinearSolverGraph* linearFEM = new CGLinearSolverGraph;
    solverFEM->setName("Euler Implicit");
    solverFEM->f_rayleighStiffness.setValue(0.01);
    solverFEM->f_rayleighMass.setValue(1);

    solverFEM->setName("Conjugate Gradient");
    linearFEM->f_maxIter.setValue(20); //iteration maxi for the CG
    linearFEM->f_smallDenominatorThreshold.setValue(0.000001);
    linearFEM->f_tolerance.setValue(0.001);

    torusFEM->addObject(solverFEM);
    torusFEM->addObject(linearFEM);


    MeshLoader* loaderFEM = new MeshLoader;
    loaderFEM->load(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh").c_str());
    torusFEM->addObject(loaderFEM);

    MeshTopology* meshTorusFEM = new MeshTopology;
    torusFEM->addObject(meshTorusFEM);

    MechanicalObject3* dofFEM = new MechanicalObject3; dofFEM->setName("FEM Object");
    dofFEM->setTranslation(2.5,0,0);
    dofFEM->setRotation(90,0,0);
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
    Node* FEMVisualNode = getSimulation()->newNode("Visu");
    torusFEM->addChild(FEMVisualNode);


    OglModel* visualFEM = new OglModel;
    visualFEM->setName("visual");
    visualFEM->load(sofa::helper::system::DataRepository.getFile("mesh/torus.obj"),"","");
    visualFEM->setColor("red");
    visualFEM->setTranslation(2.5,0,0);
    visualFEM->setRotation(90,0,0);
    FEMVisualNode->addObject(visualFEM);

    BarycentricMapping3_to_Ext3* mappingFEM = new BarycentricMapping3_to_Ext3(dofFEM, visualFEM);
    mappingFEM->setName("Mapping Visual");
    FEMVisualNode->addObject(mappingFEM);


    //Node COLLISION
    Node* FEMCollisionNode = getSimulation()->newNode("Collision");
    torusFEM->addChild(FEMCollisionNode);


    MeshLoader* loaderFEM_surf = new MeshLoader;
    loaderFEM_surf->load(sofa::helper::system::DataRepository.getFile("mesh/torus_for_collision.obj").c_str());
    FEMCollisionNode->addObject(loaderFEM_surf);

    MeshTopology* meshTorusFEM_surf= new MeshTopology;
    FEMCollisionNode->addObject(meshTorusFEM_surf);

    MechanicalObject3* dofFEM_surf = new MechanicalObject3;  dofFEM_surf->setName("Collision Object FEM");
    dofFEM_surf->setTranslation(2.5,0,0);
    dofFEM_surf->setRotation(90,0,0);
    FEMCollisionNode->addObject(dofFEM_surf);

    TriangleModel* triangleFEM = new TriangleModel; triangleFEM->setName("TriangleCollision FEM");
    FEMCollisionNode->addObject(triangleFEM);

    BarycentricMechanicalMapping3_to_3* mechaMappingFEM = new BarycentricMechanicalMapping3_to_3(dofFEM, dofFEM_surf);
    FEMCollisionNode->addObject(mechaMappingFEM);


    //************************************
    //Torus Spring
    Node* torusSpring = getSimulation()->newNode("Spring");
    chain->addChild(torusSpring);


    EulerImplicitSolver* solverSpring = new EulerImplicitSolver;
    CGLinearSolverGraph* linearSpring = new CGLinearSolverGraph;
    solverSpring->setName("Euler Implicit");
    solverSpring->f_rayleighStiffness.setValue(0.01);
    solverSpring->f_rayleighMass.setValue(1);

    linearSpring->setName("Conjugate Gradient");
    linearSpring->f_maxIter.setValue(20); //iteration maxi for the CG
    linearSpring->f_smallDenominatorThreshold.setValue(0.000001);
    linearSpring->f_tolerance.setValue(0.001);

    torusSpring->addObject(solverSpring);
    torusSpring->addObject(linearSpring);

    MeshLoader* loaderSpring = new MeshLoader;
    loaderSpring->load(sofa::helper::system::DataRepository.getFile("mesh/torus_low_res.msh").c_str());
    torusSpring->addObject(loaderSpring);
    loaderSpring->init();

    MeshTopology* meshTorusSpring = new MeshTopology;
    torusSpring->addObject(meshTorusSpring);

    MechanicalObject3* dofSpring = new MechanicalObject3; dofSpring->setName("Spring Object");
    dofSpring->setTranslation(5,0.0,0.0);
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
    Node* SpringVisualNode = getSimulation()->newNode("Visu");
    torusSpring->addChild(SpringVisualNode);


    OglModel* visualSpring = new OglModel;
    visualSpring->setName("visual");
    visualSpring->load(sofa::helper::system::DataRepository.getFile("mesh/torus.obj"),"","");
    visualSpring->setColor("green");
    visualSpring->setTranslation(5,0.0,0.0);
    SpringVisualNode->addObject(visualSpring);

    BarycentricMapping3_to_Ext3* mappingSpring = new BarycentricMapping3_to_Ext3(dofSpring, visualSpring);
    mappingSpring->setName("Mapping Visual");
    SpringVisualNode->addObject(mappingSpring);


    //Node COLLISION
    Node* SpringCollisionNode = getSimulation()->newNode("Collision");
    torusSpring->addChild(SpringCollisionNode);


    MeshLoader* loaderSpring_surf = new MeshLoader;
    loaderSpring_surf->load(sofa::helper::system::DataRepository.getFile("mesh/torus_for_collision.obj").c_str());
    SpringCollisionNode->addObject(loaderSpring_surf);

    MeshTopology* meshTorusSpring_surf= new MeshTopology;
    SpringCollisionNode->addObject(meshTorusSpring_surf);

    MechanicalObject3* dofSpring_surf = new MechanicalObject3; dofSpring_surf->setName("Collision Object Spring");
    dofSpring_surf->setTranslation(5,0.0,0.0);
    SpringCollisionNode->addObject(dofSpring_surf);

    TriangleModel* triangleSpring = new TriangleModel; triangleSpring->setName("TriangleCollision Spring");
    SpringCollisionNode->addObject(triangleSpring);

    BarycentricMechanicalMapping3_to_3* mechaMappingSpring = new BarycentricMechanicalMapping3_to_3(dofSpring, dofSpring_surf);
    SpringCollisionNode->addObject(mechaMappingSpring);


    //************************************
    //Torus FFD
    Node* torusFFD = getSimulation()->newNode("FFD");
    chain->addChild(torusFFD);


    EulerImplicitSolver* solverFFD = new EulerImplicitSolver;
    CGLinearSolverGraph* linearFFD = new CGLinearSolverGraph;
    solverFFD->setName("Euler Implicit");
    solverFFD->f_rayleighStiffness.setValue(0.01);
    solverFFD->f_rayleighMass.setValue(1);

    linearFFD->setName("Conjugate Gradient");
    linearFFD->f_maxIter.setValue(20); //iteration maxi for the CG
    linearFFD->f_smallDenominatorThreshold.setValue(0.000001);
    linearFFD->f_tolerance.setValue(0.001);

    torusFFD->addObject(solverFFD);
    torusFFD->addObject(linearFFD);

    MechanicalObject3* dofFFD = new MechanicalObject3; dofFFD->setName("FFD Object");
    dofFFD->setTranslation(7.5,0,0);
    dofFFD->setRotation(90,0,0);
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
    Node* FFDVisualNode = getSimulation()->newNode("Visu");
    torusFFD->addChild(FFDVisualNode);


    OglModel* visualFFD = new OglModel;
    visualFFD->setName("visual");
    visualFFD->load(sofa::helper::system::DataRepository.getFile("mesh/torus.obj"),"","");
    visualFFD->setColor("yellow");
    visualFFD->setTranslation(7.5,0,0);
    FFDVisualNode->addObject(visualFFD);

    BarycentricMapping3_to_Ext3* mappingFFD = new BarycentricMapping3_to_Ext3(dofFFD, visualFFD);
    mappingFFD->setName("Mapping Visual");
    FFDVisualNode->addObject(mappingFFD);


    //Node COLLISION
    Node* FFDCollisionNode = getSimulation()->newNode("Collision");
    torusFFD->addChild(FFDCollisionNode);

    MeshLoader* loaderFFD_surf = new MeshLoader;
    loaderFFD_surf->load(sofa::helper::system::DataRepository.getFile("mesh/torus_for_collision.obj").c_str());
    FFDCollisionNode->addObject(loaderFFD_surf);

    MeshTopology* meshTorusFFD_surf= new MeshTopology;
    FFDCollisionNode->addObject(meshTorusFFD_surf);

    MechanicalObject3* dofFFD_surf = new MechanicalObject3; dofFFD_surf->setName("Collision Object FFD");
    dofFFD_surf->setTranslation(7.5,0,0);
    FFDCollisionNode->addObject(dofFFD_surf);

    TriangleModel* triangleFFD = new TriangleModel;  triangleFFD->setName("TriangleCollision FFD");
    FFDCollisionNode->addObject(triangleFFD);

    BarycentricMechanicalMapping3_to_3* mechaMappingFFD = new BarycentricMechanicalMapping3_to_3(dofFFD, dofFFD_surf);
    FFDCollisionNode->addObject(mechaMappingFFD);

    //************************************
    //Torus Rigid
    Node* torusRigid = getSimulation()->newNode("Rigid");
    chain->addChild(torusRigid);

    EulerImplicitSolver* solverRigid = new EulerImplicitSolver;
    CGLinearSolverGraph* linearRigid = new CGLinearSolverGraph;
    solverRigid->setName("Euler Implicit");
    solverRigid->f_rayleighStiffness.setValue(0.01);
    solverRigid->f_rayleighMass.setValue(1);

    solverRigid->setName("Conjugate Gradient");
    linearRigid->f_maxIter.setValue(20); //iteration maxi for the CG
    linearRigid->f_smallDenominatorThreshold.setValue(0.000001);
    linearRigid->f_tolerance.setValue(0.001);

    torusRigid->addObject(solverRigid);
    torusRigid->addObject(linearRigid);

    MechanicalObjectRigid3* dofRigid = new MechanicalObjectRigid3; dofRigid->setName("Rigid Object");
    dofRigid->setTranslation(10,0,0);
    torusRigid->addObject(dofRigid);

    UniformMassRigid3* uniMassRigid = new UniformMassRigid3;
    uniMassRigid->setTotalMass(1); //the whole object will have 5 as given mass
    torusRigid->addObject(uniMassRigid);

    //Node VISUAL
    Node* RigidVisualNode = getSimulation()->newNode("Visu");
    torusRigid->addChild(RigidVisualNode);


    OglModel* visualRigid = new OglModel;
    visualRigid->setName("visual");
    visualRigid->load(sofa::helper::system::DataRepository.getFile("mesh/torus.obj"),"","");
    visualRigid->setColor("gray");
    RigidVisualNode->addObject(visualRigid);

    RigidMappingRigid3_to_Ext3* mappingRigid = new RigidMappingRigid3_to_Ext3(dofRigid, visualRigid);
    mappingRigid->setName("Mapping Visual");
    RigidVisualNode->addObject(mappingRigid);


    //Node COLLISION
    Node* RigidCollisionNode = getSimulation()->newNode("Collision");
    torusRigid->addChild(RigidCollisionNode);


    MeshLoader* loaderRigid_surf = new MeshLoader;
    loaderRigid_surf->load(sofa::helper::system::DataRepository.getFile("mesh/torus_for_collision.obj").c_str());
    RigidCollisionNode->addObject(loaderRigid_surf);

    MeshTopology* meshTorusRigid_surf= new MeshTopology;
    RigidCollisionNode->addObject(meshTorusRigid_surf);

    MechanicalObject3* dofRigid_surf = new MechanicalObject3; dofRigid_surf->setName("Collision Object Rigid");
    RigidCollisionNode->addObject(dofRigid_surf);

    TriangleModel* triangleRigid = new TriangleModel;  triangleRigid->setName("TriangleCollision Rigid");
    RigidCollisionNode->addObject(triangleRigid);

    RigidMechanicalMappingRigid3_to_3* mechaMappingRigid = new RigidMechanicalMappingRigid3_to_3(dofRigid, dofRigid_surf);
    RigidCollisionNode->addObject(mechaMappingRigid);

    return root;
}









int main(int argc, char** argv)
{
    glutInit(&argc,argv);

    std::vector<std::string> files;
    std::string simulationType="tree";

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&simulationType,'s',"simulation","type of the simulation(bgl,tree)")
    (argc,argv);

#ifdef SOFA_HAS_BOOST_KERNEL
    if (simulationType == "bgl")
        sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    else
        sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
#else
    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
#endif
    sofa::gui::GUIManager::Init(argv[0]);

    Node *root=createChainHybrid(simulationType);

    root->setAnimate(false);

    getSimulation()->init(root);


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}
