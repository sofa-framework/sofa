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
#include <sofa/defaulttype/VecTypes.h>

#include <SofaGraphComponent/Gravity.h>
#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/core/loader/MeshLoader.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaBaseMechanics/UniformMass.h>
#include <SofaBoundaryCondition/ConstantForceField.h>
#include <SofaSimpleFem/HexahedronFEMForceField.h>
#include <SofaDeformable/QuadBendingSprings.h>
#include <sofa/component/container/MeshLoader.h>
#include <SofaBoundaryCondition/FixedConstraint.h>

// solvers
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>

// collision pipeline
#include <SofaBaseCollision/DefaultPipeline.h>
#include <SofaBaseCollision/BruteForceDetection.h>
#include <SofaBaseCollision/NewProximityIntersection.h>
#include <SofaBaseCollision/DefaultContactManager.h>
#include <sofa/component/collision/TreeCollisionGroupManager.h>

//#include <sofa/component/typedef/Sofa_typedef.h>
#include <SofaOpenglVisual/OglModel.h>
#include <SofaBaseMechanics/BarycentricMapping.h>

#include <sofa/core/objectmodel/Context.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>

#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/glut.h>

#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/tree/TreeSimulation.h>

#include <iostream>
#include <fstream>

#ifdef SOFA_GPU_CUDA
#include <sofa/gpu/cuda/mycuda.h>
#endif

using namespace sofa::simulation::tree;
using namespace sofa::defaulttype;

using sofa::simulation::Node;
using sofa::helper::system::DataRepository;


// collision pipeline
using sofa::component::collision::DefaultPipeline;
using sofa::component::collision::BruteForceDetection;
using sofa::component::collision::NewProximityIntersection;
using sofa::component::collision::DefaultContactManager;
using sofa::component::collision::TreeCollisionGroupManager;
using sofa::component::collision::TriangleModel;

// solvers
using sofa::component::odesolver::EulerImplicitSolver;
using sofa::component::linearsolver::CGLinearSolver;
using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;

// mechanical object
using sofa::component::container::MechanicalObject;
using sofa::defaulttype::StdVectorTypes;
using sofa::defaulttype::Vec;
using sofa::core::loader::MeshLoader;
using sofa::component::mass::UniformMass;
using sofa::component::forcefield::ConstantForceField;
using sofa::component::forcefield::HexahedronFEMForceField;
//using sofa::component::interactionforcefield::QuadBendingSprings;

// visual
using sofa::component::visualmodel::OglModel;
using sofa::component::mapping::BarycentricMapping;
using sofa::core::Mapping;
using sofa::core::behavior::MechanicalState;
using sofa::core::State;
using sofa::defaulttype::ExtVectorTypes;

using sofa::component::topology::MeshTopology;
using sofa::component::projectiveconstraintset::FixedConstraint;

int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    sofa::helper::parse("This is a SOFA application.")
    (argc,argv);

    sofa::gui::initMain();
    sofa::gui::GUIManager::Init(argv[0]);

    // The graph root node : gravity already exists in a GNode by default
    GNode* groot = new GNode;
    groot->setName( "root" );
    groot->setGravityInWorld( Vec3dTypes::Coord(0,0,0) );
    groot->setDt(0.02);

    /*
     * collision pipeline: instead of calling ObjectCreator
     */
    // collision pipeline
    DefaultPipeline* collisionPipeline = new DefaultPipeline;
    collisionPipeline->setName("Collision Pipeline");
    groot->addObject(collisionPipeline);

    // collision detection system
    BruteForceDetection* collisionDetection = new BruteForceDetection;
    collisionDetection->setName("Collision Detection");
    groot->addObject(collisionDetection);

    // component to detection intersection
    NewProximityIntersection* detectionProximity = new NewProximityIntersection;
    detectionProximity->setName("Detection Proximity");
    detectionProximity->setAlarmDistance(0.3);
    detectionProximity->setContactDistance(0.2);
    groot->addObject(detectionProximity);

    // contact manager
    DefaultContactManager* contactManager = new DefaultContactManager;
    contactManager->setName("Contact Manager");
    contactManager->setDefaultResponseType("default");
    groot->addObject(contactManager);

    // tree collision group
    TreeCollisionGroupManager* collisionGroupManager = new TreeCollisionGroupManager;
    collisionGroupManager->setName("Collision Group Manager");
    groot->addObject(collisionGroupManager);



    /*
     * Sub nodes: DRE
     */
    GNode* dreNode = new GNode;
    dreNode->setName("DRE");


    GNode* cylNode = new GNode;
    cylNode->setName("Cylinder");


    // solvers
    typedef CGLinearSolver<GraphScatteredMatrix, GraphScatteredVector> CGLinearSolverGraph;
    EulerImplicitSolver* implicitSolver = new EulerImplicitSolver;
    CGLinearSolverGraph* cgLinearSolver = new CGLinearSolverGraph;

    implicitSolver->setName("eulerImplicitSolver");
    implicitSolver->f_rayleighStiffness.setValue(0.01);
    //implicitSolver->f_rayleighMass.setValue(0.1);
    implicitSolver->f_printLog = false;
    cgLinearSolver->setName("cgLinearSolver");
    cgLinearSolver->f_maxIter.setValue(25);
    cgLinearSolver->f_tolerance.setValue(1.0e-9);
    cgLinearSolver->f_smallDenominatorThreshold.setValue(1.0e-9);




    // sparse grid topology
    sofa::component::topology::SparseGridTopology* sparseGridTopology = new sofa::component::topology::SparseGridTopology;
    sparseGridTopology->setName("SparseGrid Topology");
    std::string topologyFilename = "mesh/truthcylinder1.obj";
    sparseGridTopology->load(topologyFilename.c_str());
    sparseGridTopology->setN(Vec<3,int>(8, 6, 6));


    // mechanical object
    typedef MechanicalObject< Vec3dTypes > MechanicalObject3d;
    MechanicalObject3d* mechanicalObject = new MechanicalObject3d;
    mechanicalObject->setTranslation(0,0,0);
    mechanicalObject->setRotation(0,0,0);
    mechanicalObject->setScale(1,1,1);


    // mass
    typedef UniformMass< Vec3dTypes,double > UniformMass3d;
    UniformMass3d* uniformMass = new UniformMass3d;
    uniformMass->setTotalMass(5);


    // hexahedron fem forcefield
    typedef HexahedronFEMForceField< Vec3dTypes > HexahedronFEMForceField3d;
    HexahedronFEMForceField3d* hexaFEMFF = new HexahedronFEMForceField3d;
    hexaFEMFF->setName("HexahedronFEM Forcefield");
    hexaFEMFF->setMethod(HexahedronFEMForceField3d::POLAR);
    hexaFEMFF->setPoissonRatio(0.3);
    hexaFEMFF->setYoungModulus(250);


    // quad bending springs
    typedef sofa::component::interactionforcefield::QuadBendingSprings< Vec3dTypes > QuadBendingSprings3d;
    QuadBendingSprings3d* quadBendingSprings = new QuadBendingSprings3d;
    quadBendingSprings->setName("QuadBending springs");
    quadBendingSprings->setStiffness(1000);
    quadBendingSprings->setDamping(1);
    quadBendingSprings->setObject1(mechanicalObject);


    // fixed constraint
    typedef FixedConstraint< StdVectorTypes<Vec<3,double>,Vec<3,double>,double> > FixedConstraint3d;
    FixedConstraint3d* fixedConstraints = new FixedConstraint3d;
    fixedConstraints->setName("Box Constraints");
    fixedConstraints->addConstraint(0);
    fixedConstraints->addConstraint(1); fixedConstraints->addConstraint(2); fixedConstraints->addConstraint(6); fixedConstraints->addConstraint(12); fixedConstraints->addConstraint(17); fixedConstraints->addConstraint(21); fixedConstraints->addConstraint(22);
    fixedConstraints->addConstraint(24); fixedConstraints->addConstraint(25); fixedConstraints->addConstraint(26); fixedConstraints->addConstraint(30); fixedConstraints->addConstraint(36); fixedConstraints->addConstraint(41); fixedConstraints->addConstraint(46); fixedConstraints->addConstraint(47);
    fixedConstraints->addConstraint(50); fixedConstraints->addConstraint(51); fixedConstraints->addConstraint(52); fixedConstraints->addConstraint(56); fixedConstraints->addConstraint(62); fixedConstraints->addConstraint(68); fixedConstraints->addConstraint(73); fixedConstraints->addConstraint(74);
    fixedConstraints->addConstraint(77); fixedConstraints->addConstraint(78); fixedConstraints->addConstraint(79); fixedConstraints->addConstraint(83); fixedConstraints->addConstraint(89); fixedConstraints->addConstraint(95); fixedConstraints->addConstraint(100); fixedConstraints->addConstraint(101);
    fixedConstraints->addConstraint(104); fixedConstraints->addConstraint(105); fixedConstraints->addConstraint(106); fixedConstraints->addConstraint(110); fixedConstraints->addConstraint(116); fixedConstraints->addConstraint(122); fixedConstraints->addConstraint(127); fixedConstraints->addConstraint(128);
    fixedConstraints->addConstraint(131); fixedConstraints->addConstraint(132); fixedConstraints->addConstraint(133); fixedConstraints->addConstraint(137); fixedConstraints->addConstraint(143); fixedConstraints->addConstraint(149); fixedConstraints->addConstraint(154); fixedConstraints->addConstraint(155);
    fixedConstraints->addConstraint(158); fixedConstraints->addConstraint(159); fixedConstraints->addConstraint(160); fixedConstraints->addConstraint(164); fixedConstraints->addConstraint(170); fixedConstraints->addConstraint(175); fixedConstraints->addConstraint(180); fixedConstraints->addConstraint(181);
    fixedConstraints->addConstraint(184); fixedConstraints->addConstraint(185); fixedConstraints->addConstraint(186); fixedConstraints->addConstraint(190); fixedConstraints->addConstraint(196); fixedConstraints->addConstraint(201); fixedConstraints->addConstraint(206); fixedConstraints->addConstraint(205);


    // visual node
    GNode* cylVisualNode = new GNode;
    cylVisualNode->setName("Cylinder Visual");

    OglModel* cylOglModel = new OglModel;
    cylOglModel->setName("Visual");
    std::string visualFilename = "mesh/truthcylinder1.obj";
    cylOglModel->setFilename(DataRepository.getFile(visualFilename).c_str());
    cylOglModel->setColor("red");


    typedef BarycentricMapping< Vec3dTypes, ExtVec3fTypes > BarycentricMapping3d_to_Ext3f;
    BarycentricMapping3d_to_Ext3f* barycentricMapping = new BarycentricMapping3d_to_Ext3f(mechanicalObject, cylOglModel);
    barycentricMapping->setName("Barycentric");
    //barycentricMapping->setPathInputObject("../..");
    //barycentricMapping->setPathOutputObject("Visual");




    // collision node
    GNode* cylCollisionNode = new GNode;
    cylCollisionNode->setName("Cylinder Collision");

    sofa::component::container::MeshLoader* cylSurfMeshLoader = new sofa::component::container::MeshLoader;
    std::string collisionFilename = "mesh/truthcylinder1.msh";
    cylSurfMeshLoader->setFilename(DataRepository.getFile(collisionFilename).c_str());

    MeshTopology* cylSurfaceTopology = new MeshTopology;


    MechanicalObject3d* cylSurfMechanicalObject = new MechanicalObject3d;

    TriangleModel* triangleModel = new TriangleModel;

    typedef BarycentricMapping< Vec3dTypes, Vec3dTypes > BarycentricMechanicalMapping3d_to_3d;
    BarycentricMechanicalMapping3d_to_3d* cylSurfBarycentricMapping = new BarycentricMechanicalMapping3d_to_3d(mechanicalObject, cylSurfMechanicalObject);
    //cylSurfBarycentricMapping->setPathInputObject("../..");
    //cylSurfBarycentricMapping->setPathOutputObject("..");

    cylVisualNode->addObject(cylOglModel);
    cylVisualNode->addObject(barycentricMapping);


    cylCollisionNode->addObject(cylSurfMeshLoader);
    cylCollisionNode->addObject(cylSurfaceTopology);
    cylCollisionNode->addObject(cylSurfMechanicalObject);
    cylCollisionNode->addObject(triangleModel);
    cylCollisionNode->addObject(cylSurfBarycentricMapping);
    quadBendingSprings->setObject2(cylSurfMechanicalObject);


    cylNode->addObject(implicitSolver);
    cylNode->addObject(cgLinearSolver);
    cylNode->addObject(sparseGridTopology);
    cylNode->addObject(mechanicalObject);
    cylNode->addObject(uniformMass);
    cylNode->addObject(hexaFEMFF);
    cylNode->addObject(quadBendingSprings);
    cylNode->addObject(fixedConstraints);

    cylNode->addChild(cylVisualNode);
    cylNode->addChild(cylCollisionNode);

    dreNode->addChild(cylNode);
    groot->addChild(dreNode);
#ifdef SOFA_GPU_CUDA
    sofa::gpu::cuda::mycudaInit();
#endif


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
    sofa::gui::GUIManager::MainLoop(groot);

    return 0;

}
