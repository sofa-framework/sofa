
#include <sofa/helper/ArgumentParser.h>
#include <sofa/gui/GUIManager.h>


#include <sofa/component/contextobject/CoordinateSystem.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/core/objectmodel/Context.h>
#include <sofa/gui/SofaGUI.h>

//Including components for collision detection
#include <sofa/component/collision/DefaultPipeline.h>
#include <sofa/component/collision/DefaultContactManager.h>
#include <sofa/component/collision/BruteForceDetection.h>
#include <sofa/component/collision/NewProximityIntersection.h>
#include <sofa/component/collision/TriangleModel.h>

//Including component for topological description of the objects
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/RegularGridTopology.h>

//Including Solvers
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>



#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/TransformationVisitor.h>
#include <sofa/helper/system/glut.h>

#include <sofa/helper/system/SetDirectory.h>


//SOFA_HAS_BOOST_KERNEL to define in chainHybrid.pro

#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/bgl/BglSimulation.h>
#include <sofa/component/collision/BglCollisionGroupManager.h>
#include <sofa/component/collision/SphereModel.h>



//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>



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

    EulerImplicitSolver* solver = new EulerImplicitSolver;
    CGLinearSolverGraph* linear = new CGLinearSolverGraph;
    solver->setName("Euler Implicit");
    solver->f_rayleighStiffness.setValue(0.01);
    solver->f_rayleighMass.setValue(1);

    linear->setName("Conjugate Gradient");
    linear->f_maxIter.setValue(20); //iteration maxi for the CG
    linear->f_smallDenominatorThreshold.setValue(0.000001);
    linear->f_tolerance.setValue(0.001);

    root->addObject(solver);
    root->addObject(linear);

    //--> adding component to handle groups of collision.
    //BglCollisionGroupManager* collisionGroupManager = new BglCollisionGroupManager;
    //collisionGroupManager->setName("Collision Group Manager");
    //root->addObject(collisionGroupManager);

    return root;
}

Node* createRegularGrid(double x, double y, double z)
{
    static unsigned int i = 1;
    std::ostringstream oss;
    oss << "regularGrid_" << i;

    Node* node = getSimulation()->newNode(oss.str() );

    RegularGridTopology* grid = new RegularGridTopology(3,3,3);
    grid->setPos(-1.5+x,1.5+x,-1.5+y,1.5+y,-1.5+z,1.5+z);
    MechanicalObject3* dof = new MechanicalObject3;

    UniformMass3* mass = new UniformMass3;
    mass->setTotalMass(100);

    HexahedronFEMForceField3* ff = new HexahedronFEMForceField3();
    ff->setYoungModulus(400);
    ff->setPoissonRatio(0.3);
    ff->setMethod(0);

    node->addObject(dof);
    node->addObject(mass);
    node->addObject(grid);
    node->addObject(ff);

    return node;
}


int main(int argc, char** argv)
{
    glutInit(&argc,argv);

    std::vector<std::string> files;

    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
//       .option(&simulationType,'s',"simulation","type of the simulation(bgl,tree)")
    (argc,argv);

    sofa::simulation::setSimulation(new sofa::simulation::bgl::BglSimulation());
    sofa::gui::GUIManager::Init(argv[0]);

    Node *root=createRoot();
    Node* grid1 = createRegularGrid(-3,0,0);
    Node* grid2 = createRegularGrid(3,0,0);
    root->addChild(grid1);
    root->addChild(grid2);

    MechanicalObject3* subsetDof = new MechanicalObject3;
    SubsetMultiMappingVec3d_to_Vec3d* subsetMultiMapping = new SubsetMultiMappingVec3d_to_Vec3d();
    MechanicalObject3* input1 = dynamic_cast<MechanicalObject3*>(grid2->getMechanicalState());
    MechanicalObject3* input2 = dynamic_cast<MechanicalObject3*>(grid1->getMechanicalState());
    subsetMultiMapping->addInputModel( input1 );
    subsetMultiMapping->addInputModel( input2 );
    subsetMultiMapping->addOutputModel( subsetDof );

    subsetMultiMapping->addPoint( input1, 21);
    subsetMultiMapping->addPoint( input1, 18);
    subsetMultiMapping->addPoint( input1, 9);
    subsetMultiMapping->addPoint( input1, 12);

    subsetMultiMapping->addPoint( input2, 11);
    subsetMultiMapping->addPoint( input2, 20);
    subsetMultiMapping->addPoint( input2, 14);
    subsetMultiMapping->addPoint( input2, 23);


    MeshTopology* topology = new MeshTopology;

    topology->addHexa(4,2,3,6,5,1,0,7);

    HexahedronFEMForceField3* ff = new HexahedronFEMForceField3();
    ff->setYoungModulus(400);
    ff->setPoissonRatio(0.3);
    ff->setMethod(0);




    Node* multiParentsNode = getSimulation()->newNode("MultiParents");
    multiParentsNode->addObject(topology);
    multiParentsNode->addObject(subsetDof);
    multiParentsNode->addObject(subsetMultiMapping);

    multiParentsNode->addObject(ff);



    grid1->addChild(multiParentsNode);
    grid2->addChild(multiParentsNode);
    root->setAnimate(false);

    getSimulation()->init(root);


    //=======================================
    // Run the main loop
    sofa::gui::GUIManager::MainLoop(root);

    return 0;
}
