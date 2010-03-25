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
#include <sofa/gui/GUIManager.h>


#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/gui/SofaGUI.h>

#include <sofa/component/constraint/FixedConstraint.h>

#include <sofa/component/forcefield/ConstantForceField.h>
#include <sofa/component/forcefield/JointSpringForceField.h>
//Including components for collision detection
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/TriangleModel.h>

//Including component for topological description of the objects
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/container/MeshLoader.h>

//Including Solvers
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>

#include <sofa/component/visualmodel/OglModel.h>

#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/TransformationVisitor.h>
#include <sofa/helper/system/glut.h>

#include <sofa/helper/system/SetDirectory.h>

#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/component/collision/BglCollisionGroupManager.h>
#include <sofa/component/collision/SphereModel.h>


using sofa::component::visualmodel::OglModel;

using namespace sofa::simulation;
using namespace sofa::component::forcefield;
using namespace sofa::component::collision;
using namespace sofa::component::topology;
using sofa::component::container::MeshLoader;
using sofa::component::odesolver::EulerImplicitSolver;
using sofa::component::odesolver::EulerSolver;
using sofa::component::linearsolver::CGLinearSolver;
using sofa::component::linearsolver::GraphScatteredMatrix;
using sofa::component::linearsolver::GraphScatteredVector;
typedef CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> CGLinearSolverGraph;


//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
// ---------------------------------------------------------------------
// ---
// ---------------------------------------------------------------------

Node* createRoot()
{
    Node* root = getSimulation()->newNode("root");
    root->setGravityInWorld( Coord3(0,0,0) );
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
    //BglCollisionGroupManager* collisionGroupManager = new BglCollisionGroupManager;
    //collisionGroupManager->setName("Collision Group Manager");
    //root->addObject(collisionGroupManager);

    return root;
}

Node* createCube(double dx, double dy, double dz, bool implicit = false)
{
    static int i = 1;
    std::ostringstream oss;
    oss << "cube_" << i++;
    Node* cube_node = getSimulation()->newNode(oss.str());


    if ( implicit )
    {

        EulerImplicitSolver* solver = new EulerImplicitSolver;
        CGLinearSolverGraph* linear = new CGLinearSolverGraph;
        solver->setName("Euler Implicit");
        solver->f_rayleighStiffness.setValue(0.01);
        solver->f_rayleighMass.setValue(1);

        linear->setName("Conjugate Gradient");
        linear->f_maxIter.setValue(20); //iteration maxi for the CG
        linear->f_smallDenominatorThreshold.setValue(0.000001);
        linear->f_tolerance.setValue(0.001);

        cube_node->addObject(solver);
        cube_node->addObject(linear);

    }
    else
    {

        EulerSolver* solver = new EulerSolver;
        solver->setName("Euler Explicit");
        solver->f_printLog.setValue(false);
        cube_node->addObject(solver);

    }

    MechanicalObject3* DOF = new MechanicalObject3;
    cube_node->addObject(DOF);
    DOF->setName("cube");
    DOF->setTranslation(dx,dy,dz);

    CubeTopology* cubeTopology = new CubeTopology(2,2,2);
    cubeTopology->setPos(-1,1,-1,1,-1,1);
    cube_node->addObject(cubeTopology);


    TriangleFEMForceField3* triangleFEM = new TriangleFEMForceField3;
    triangleFEM->setName("FEM");
    //triangleFEM->setComputeGlobalMatrix(false);
    triangleFEM->setMethod(0);
    triangleFEM->setPoisson(0.3);
    triangleFEM->setYoung(500);
    cube_node->addObject(triangleFEM);


    UniformMass3* uniMassCube = new UniformMass3;
    uniMassCube->setTotalMass(1);
    cube_node->addObject(uniMassCube);

    return cube_node;
}



int main( int argc, char** argv )
{
    glutInit(&argc,argv);

    std::vector<std::string> files;
    bool implicit=true;
    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&implicit,'i',"implicit","Implicit Integration Scheme")
    (argc,argv);

    sofa::simulation::setSimulation( new sofa::simulation::bgl::BglSimulation() );
    sofa::gui::GUIManager::Init(argv[0]);

    Node *root= createRoot();


    Node* cube1 = createCube( 0,0,0,   implicit  );
    Node* cube2 = createCube( 10,0,0,  implicit );
    Node* cube3 = createCube( 0,0,10,  implicit );
    Node* cube4 = createCube( 10,0,10, implicit);
    Node* MultiParentsNode = getSimulation()->newNode("MultiParentsNode");

    MechanicalObject3* dofMultiMapping = new MechanicalObject3; dofMultiMapping->setName("Center Of Mass");

    MultiParentsNode->addObject(dofMultiMapping);

    CenterofMassMechanicalMultiMappingVec3d_to_Vec3d* multiMappingCOM = new CenterofMassMechanicalMultiMappingVec3d_to_Vec3d();
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube1->getMechanicalState()) );
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube2->getMechanicalState()) );
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube3->getMechanicalState()) );
    multiMappingCOM->addInputModel( dynamic_cast<MechanicalObject3*>(cube4->getMechanicalState()) );
    multiMappingCOM->addOutputModel(dofMultiMapping);


    MultiParentsNode->addObject(multiMappingCOM);

    ConstantForceField3* constantFF = new ConstantForceField3();
    constantFF->setForce( 0,MechanicalObject3::Deriv(0,10,0) );
    MultiParentsNode->addObject(constantFF) ;
    MultiParentsNode->addObject( new SphereModel);

    cube1->addChild(MultiParentsNode);
    cube2->addChild(MultiParentsNode);
    cube3->addChild(MultiParentsNode);
    cube4->addChild(MultiParentsNode);

    root->addChild( cube1 );
    root->addChild( cube2 );
    root->addChild( cube3 );
    root->addChild( cube4 );






    root->setAnimate(false);


    getSimulation()->init(root);


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}

