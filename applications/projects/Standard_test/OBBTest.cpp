#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#ifdef SOFA_HAVE_BGL
#include <sofa/simulation/bgl/BglSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/component/init.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
#include "../../../applications/tutorials/objectCreator/ObjectCreator.h"

#include <plugins/Flexible/deformationMapping/ExtensionMapping.h>
#include <plugins/Flexible/deformationMapping/DistanceMapping.h>

#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/TreeCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/collision/MeshIntTool.h>

struct TestOBB : public ::testing::Test{
    typedef sofa::defaulttype::Vec3d Vec3d;

    /**
      *\brief Rotates around x axis vectors x,y and z which here is a frame.
      */
    static void rotx(double ax,Vec3d & x,Vec3d & y,Vec3d & z);
    static void roty(double ay,Vec3d & x,Vec3d & y,Vec3d & z);
    static void rotz(double ay,Vec3d & x,Vec3d & y,Vec3d & z);


    /**
      *\brief Makes up an OBBModel containing just one OBB. angles and order are the rotations used to make up this OBB.
      *
      *\param p the center of the OBB
      *\param angles it is of size 3 and contains the rotations around axes, i.e., angles[0] contains rotation around x axis etc...
      *\param order it is the order we rotate, i.e, if we want to rotate first around z axis, then x axis and then y axis order will be {2,0,1}
      *\param v it is the velocity of the OBB
      *\param extents it contains half-extents of the OBB
      *\param father it is a node that will contain the returned OBBModel
      */
    static sofa::component::collision::OBBModel::SPtr makeOBB(const Vec3d & p,const double * angles,const int * order,const Vec3d & v,const Vec3d & extents,
                                                                sofa::simulation::Node::SPtr & father);


    bool faceVertex();
    bool vertexVertex();
    bool faceFace();
    bool faceEdge();
    bool edgeEdge();
    bool edgeVertex();


//    sofa::component::collision::OBB movingOBB;
//    sofa::component::collision::OBB staticOBB;
};


struct TestCapOBB  : public ::testing::Test{
    typedef sofa::defaulttype::Vec3d Vec3d;

    static sofa::component::collision::CapsuleModel::SPtr makeCap(const Vec3d & p0,const Vec3d & p1,double radius,const Vec3d & v,
                                                                  sofa::simulation::Node::SPtr & father);

    bool faceVertex();
    bool faceEdge();
    bool edgeVertex();
    bool edgeEdge();
    bool vertexVertex();
    bool vertexEdge();
};

struct TestSphereOBB : public ::testing::Test{
    typedef sofa::defaulttype::Vec3d Vec3d;

    static sofa::component::collision::SphereModel::SPtr makeSphere(const Vec3d & center,double radius,const Vec3d & v,
                                                                    sofa::simulation::Node::SPtr & father);

    bool vertex();
    bool edge();
    bool face();
};


struct TestTriOBB : public ::testing::Test{
    typedef sofa::defaulttype::Vec3d Vec3d;

    static sofa::component::collision::TriangleModel::SPtr makeTri(const Vec3d & p0,const Vec3d & p1,const Vec3d & p2,const Vec3d &v, sofa::simulation::Node::SPtr &father);

    bool faceVertex();
    bool vertexVertex();
    bool faceFace();
    bool faceEdge();
    bool edgeFace();
    bool edgeEdge();
    bool edgeEdge2();
    bool edgeVertex();
    bool vertexFace();
    bool vertexEdge();
};

sofa::component::collision::SphereModel::SPtr TestSphereOBB::makeSphere(const Vec3d & center,double radius,const Vec3d & v,
                                                                   sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr sph = father->createChild("cap");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3d::SPtr sphDOF = New<MechanicalObject3d>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    sphDOF->resize(1);
    Data<MechanicalObject3d::VecCoord> & dpositions = *sphDOF->write( sofa::core::VecId::position() );
    MechanicalObject3d::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = center;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3d::VecDeriv> & dvelocities = *sphDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3d::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();

    sph->addObject(sphDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::SphereModel::SPtr sphCollisionModel = New<sofa::component::collision::SphereModel >();
    sph->addObject(sphCollisionModel);


    //editting the OBBModel
    sphCollisionModel->init();
    Data<sofa::component::collision::SphereModel::VecReal> & dVecReal = sphCollisionModel->radius;
    sofa::component::collision::CapsuleModel::VecReal & vecReal = *(dVecReal.beginEdit());

    vecReal[0] = radius;

    dVecReal.endEdit();

    return sphCollisionModel;
}

sofa::component::collision::CapsuleModel::SPtr TestCapOBB::makeCap(const Vec3d & p0,const Vec3d & p1,double radius,const Vec3d & v,
                                                                   sofa::simulation::Node::SPtr & father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr cap = father->createChild("cap");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3d::SPtr capDOF = New<MechanicalObject3d>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    capDOF->resize(2);
    Data<MechanicalObject3d::VecCoord> & dpositions = *capDOF->write( sofa::core::VecId::position() );
    MechanicalObject3d::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3d::VecDeriv> & dvelocities = *capDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3d::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    dvelocities.endEdit();

    cap->addObject(capDOF);

    //creating a topology necessary for capsule
    sofa::component::topology::MeshTopology::SPtr bmt = New<sofa::component::topology::MeshTopology>();
    bmt->addEdge(0,1);
    cap->addObject(bmt);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::CapsuleModel::SPtr capCollisionModel = New<sofa::component::collision::CapsuleModel >();
    cap->addObject(capCollisionModel);


    //editting the OBBModel
    capCollisionModel->init();
    Data<sofa::component::collision::CapsuleModel::VecReal> & dVecReal = capCollisionModel->writeRadii();
    sofa::component::collision::CapsuleModel::VecReal & vecReal = *(dVecReal.beginEdit());

    vecReal[0] = radius;

    dVecReal.endEdit();

    return capCollisionModel;
}


sofa::component::collision::TriangleModel::SPtr TestTriOBB::makeTri(const Vec3d & p0,const Vec3d & p1,const Vec3d & p2,const Vec3d & v, sofa::simulation::Node::SPtr &father){
    //creating node containing TriangleModel
    sofa::simulation::Node::SPtr tri = father->createChild("tri");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObject3d::SPtr triDOF = New<MechanicalObject3d>();

    //editing DOF related to the TriangleModel to be created, size is 3 (3 points) because it contains just one Triangle
    triDOF->resize(3);
    Data<MechanicalObject3d::VecCoord> & dpositions = *triDOF->write( sofa::core::VecId::position() );
    MechanicalObject3d::VecCoord & positions = *dpositions.beginEdit();

    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = p0;
    positions[1] = p1;
    positions[2] = p2;

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObject3d::VecDeriv> & dvelocities = *triDOF->write( sofa::core::VecId::velocity() );

    MechanicalObject3d::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    velocities[1] = v;
    velocities[2] = v;

    dvelocities.endEdit();

    tri->addObject(triDOF);

    //creating a topology necessary for capsule
    sofa::component::topology::MeshTopology::SPtr bmt = New<sofa::component::topology::MeshTopology>();
    bmt->addTriangle(0,1,2);
    tri->addObject(bmt);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::TriangleModel::SPtr triCollisionModel = New<sofa::component::collision::TriangleModel >();
    tri->addObject(triCollisionModel);


    //editting the OBBModel
    triCollisionModel->init();

    return triCollisionModel;
}

//sofa::simulation::Node::SPtr TestOBB::createScene()
//{
//    sofa::simulation::Node::SPtr groot = New<sofa::simulation::tree::GNode>();
////    // The graph root node
////    std::cout<<"ici0"<<std::endl;
////    sofa::simulation::setSimulation(new sofa::simulation::tree::TreeSimulation());
////    std::cout<<"ici1"<<std::endl;
////    sofa::simulation::Node::SPtr groot = sofa::simulation::getSimulation()->createNewGraph("root");
////    std::cout<<"ici2"<<std::endl;
////    groot->setGravity( Coord3(0,0,0) );
////    std::cout<<"ici3"<<std::endl;

////    // One solver for all the graph
////    sofa::component::odesolver::EulerSolver::SPtr solver = sofa::core::objectmodel::New<sofa::component::odesolver::EulerSolver>();
////    std::cout<<"ici4"<<std::endl;
////    solver->setName("solver");
////    solver->f_printLog.setValue(false);
////    std::cout<<"ici5"<<std::endl;
////    groot->addObject(solver);
////    std::cout<<"ici6"<<std::endl;

//    // One node to define the particle
////    sofa::simulation::Node::SPtr particule_node = groot.get()->createChild("particle_node");
////    // The particule, i.e, its degrees of freedom : a point with a velocity
////    MechanicalObject3::SPtr particle = sofa::core::objectmodel::New<MechanicalObject3>();
////    particle->setName("particle");
////    particule_node->addObject(particle);
////    particle->resize(1);
////    // get write access the particle positions vector
////    WriteAccessor< Data<MechanicalObject3::VecCoord> > positions = *particle->write( VecId::position() );
////    positions[0] = Coord3(0,0,0);
////    // get write access the particle velocities vector
////    WriteAccessor< Data<MechanicalObject3::VecDeriv> > velocities = *particle->write( VecId::velocity() );
////    velocities[0] = Deriv3(0,0,0);

////    // Its properties, i.e, a simple mass node
////    UniformMass3::SPtr mass = sofa::core::objectmodel::New<UniformMass3>();
////    mass->setName("mass");
////    particule_node->addObject(mass);
////    mass->setMass( 1 );

////    // Display Flags
////    sofa::component::visualmodel::VisualStyle::SPtr style = sofa::core::objectmodel::New<sofa::component::visualmodel::VisualStyle>();
////    groot->addObject(style);
////    sofa::core::visual::DisplayFlags& flags = *style->displayFlags.beginEdit();
////    flags.setShowBehaviorModels(true);
////    style->displayFlags.endEdit();

////    sofa::simulation::tree::getSimulation()->init(groot.get());
////    groot->setAnimate(false);

//    return groot;
//}

sofa::component::collision::OBBModel::SPtr TestOBB::makeOBB(const Vec3d & p,const double *angles,const int *order,const Vec3d &v,const Vec3d &extents, sofa::simulation::Node::SPtr &father){
    //creating node containing OBBModel
    sofa::simulation::Node::SPtr obb = father->createChild("obb");

    //creating a mechanical object which will be attached to the OBBModel
    MechanicalObjectRigid3::SPtr obbDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the OBBModel to be created, size is 1 because it contains just one OBB
    obbDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *obbDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    //we create a frame that we will rotate like it is specified by the parameters angles and order
    Vec3d x(1,0,0);
    Vec3d y(0,1,0);
    Vec3d z(0,0,1);

    //creating an array of functions which are the rotation so as to perform the rotations in a for loop
    typedef void (*rot)(double,Vec3d&,Vec3d&,Vec3d&);
    rot rotations[3];
    rotations[0] = &rotx;
    rotations[1] = &roty;
    rotations[2] = &rotz;

    //performing the rotations of the frame x,y,z
    for(int i = 0 ; i < 3 ; ++i)
        (*rotations[order[i]])(angles[order[i]],x,y,z);


    //we finnaly edit the positions by filling it with a RigidCoord made up from p and the rotated fram x,y,z
    positions[0] = Rigid3Types::Coord(p,Quaternion::createQuaterFromFrame(x,y,z));

    dpositions.endEdit();

    //Editting the velocity of the OBB
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *obbDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();


    obb->addObject(obbDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::OBBModel::SPtr obbCollisionModel = New<sofa::component::collision::OBBModel >();
    obb->addObject(obbCollisionModel);

    //editting the OBBModel
    obbCollisionModel->init();
    Data<sofa::component::collision::OBBModel::VecCoord> & dVecCoord = obbCollisionModel->writeExtents();
    sofa::component::collision::OBBModel::VecCoord & vecCoord = *(dVecCoord.beginEdit());

    vecCoord[0] = extents;

    dVecCoord.endEdit();

    return obbCollisionModel;
}


void TestOBB::rotx(double ax,Vec3d & x,Vec3d & y,Vec3d & z){
    Vec3d ix = Vec3d(1,0,0);

    Quaternion rotx(ix,ax);

    x = rotx.rotate(x);y = rotx.rotate(y);z = rotx.rotate(z);
}

void TestOBB::roty(double angle,Vec3d & x,Vec3d & y,Vec3d & z){
    Vec3d iy = Vec3d(0,1,0);

    Quaternion rot(iy,angle);

    x = rot.rotate(x);y = rot.rotate(y);z = rot.rotate(z);
}

void TestOBB::rotz(double angle,Vec3d & x,Vec3d & y,Vec3d & z){
    Vec3d iz = Vec3d(0,0,1);

    Quaternion rot(iz,angle);

    x = rot.rotate(x);y = rot.rotate(y);z = rot.rotate(z);
}

//vertex indexation of an OBB below :
//
//                                         7--------6
//                                        /|       /|
//                                       3--------2 |
//                                       | |      | |
//                                       | 4------|-5
//                                       |/       |/
//                                       0--------1
//
bool TestOBB::faceVertex(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //the second OBB which is moving, one OBB must move, if not, there is no collision (OBB collision algorithm is like that)
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3d(0,0,sqrt(3.0) + 0.01),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);

    //we construct OBBs from OBBModels
    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of obb0 (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb1 (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

//obb0's vertex 6 in intersection with obb1's vertex 0 (see indexation above)
bool TestOBB::vertexVertex(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3d(0,0,sqrt(3.0) + 0.01),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT)){
        return false;
    }

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    return true;
}

//obb0's face 3,2,6,7 in intersection with obb1's face 0,4,5,1 (see indexation above)
bool TestOBB::faceFace(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3d(0,1,1.01),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0.5,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0.5,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

//obb0's face 3,2,6,7 in intersection with obb1's edge 3-0
bool TestOBB::faceEdge(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3d(0,0,sqrt(2.0) + 0.01),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

//obb0's edge 6-5 in intersection with obb1's edge 3-0 (see indexation above)
bool TestOBB::edgeEdge(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3d(0,0,sqrt(2.0) + 0.01),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(-1,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(-1,0,0.01)).norm() > 1e-6)
        return false;

    return true;
}

//obb0's edge 6-5 in intersection with obb1's vertex 0 (see indexation above)
bool TestOBB::edgeVertex(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3d(0,0,sqrt(3.0) + 0.01),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    return true;
}

bool TestCapOBB::faceVertex(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel =
            TestOBB::makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3d(0,0,1 + 0.01),Vec3d(0,0,2),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Capsule cap(capmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::CapsuleIntTool::computeIntersection(cap,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

bool TestCapOBB::faceEdge(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel =
            TestOBB::makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3d(-1,0,1 + 0.01),Vec3d(1,0,1 + 0.01),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Capsule cap(capmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::CapsuleIntTool::computeIntersection(cap,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

//obb's edge 6-5 in intersection with a vertex of the capsule
bool TestCapOBB::edgeVertex(){
    //first, we create the transformation to make the first OBB
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel =
            TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3d(0,0,1 + 0.01),Vec3d(0,0,2),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Capsule cap(capmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::CapsuleIntTool::computeIntersection(cap,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

//obb's edge 6-5 in intersection parallely with the capsule
bool TestCapOBB::edgeEdge(){
    //first, we create the transformation to make the first OBB
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel =
            TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3d(-0.5,0,1 + 0.01),Vec3d(0.5,0,1 + 0.01),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Capsule cap(capmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::CapsuleIntTool::computeIntersection(cap,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


//obb's edge 6-5 in intersection parallely with the capsule
bool TestCapOBB::vertexEdge(){
    //first, we create the transformation to make the first OBB
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3d(-0.5,0,1 + 0.01),Vec3d(0.5,0,1 + 0.01),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Capsule cap(capmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::CapsuleIntTool::computeIntersection(cap,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


//obb's edge 6-5 in intersection parallely with the capsule
bool TestCapOBB::vertexVertex(){
    //first, we create the transformation to make the first OBB
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3d(0,0,1 + 0.01),Vec3d(0,0,2),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Capsule cap(capmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::CapsuleIntTool::computeIntersection(cap,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

bool TestSphereOBB::vertex(){
    //first, we create the transformation to make the first OBB
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::SphereModel::SPtr sphmodel = makeSphere(Vec3d(0,0,1 + 0.01),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Sphere sph(sphmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(sph,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestSphereOBB::edge(){
    //first, we create the transformation to make the first OBB
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,-10),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::SphereModel::SPtr sphmodel = makeSphere(Vec3d(0,0,1 + 0.01),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Sphere sph(sphmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(sph,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestSphereOBB::face(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::SphereModel::SPtr sphmodel = makeSphere(Vec3d(0,0,1 + 0.01),1,Vec3d(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Sphere sph(sphmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(sph,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

bool TestTriOBB::faceFace(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-2,-2,0.01),Vec3d(-2,2,0.01),Vec3d(2,0,0.01),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::faceEdge(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(0,-2,0.01),Vec3d(0,2,0.01),Vec3d(2,0,2),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::faceVertex(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-1),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(0,-2,2),Vec3d(0,2,2),Vec3d(0,0,0.01),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::edgeFace(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-2,-2,0.01),Vec3d(-2,2,0.01),Vec3d(2,0,0.01),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::edgeEdge(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(0,-2,0.01),Vec3d(0,2,0.01),Vec3d(2,0,2),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::edgeEdge2(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-1,0,0.01),Vec3d(1,0,0.01),Vec3d(2,0,2),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
//        return false;

    return true;
}

bool TestTriOBB::edgeVertex(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(2.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(0,0,0.01),Vec3d(1,0,2),Vec3d(-1,0,2),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
//        return false;

    return true;
}


bool TestTriOBB::vertexFace(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-2,-2,0.01),Vec3d(-2,2,0.01),Vec3d(2,0,0.01),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
//        return false;

    return true;
}


bool TestTriOBB::vertexEdge(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-1,0,0.01),Vec3d(1,0,0.01),Vec3d(2,0,2),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3d(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3d(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
//        return false;

    return true;
}

bool TestTriOBB::vertexVertex(){
    double angles[3];
    int order[3];
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = TestOBB::makeOBB(Vec3d(0,0,-sqrt(3.0)),angles,order,Vec3d(0,0,0),Vec3d(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(0,0,0.01),Vec3d(1,0,2),Vec3d(-1,0,2),Vec3d(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
//        return false;

    return true;
}

TEST_F(TestOBB, face_vertex ) { ASSERT_TRUE( faceVertex()); }
TEST_F(TestOBB, vertex_vertex ) { ASSERT_TRUE( vertexVertex()); }
TEST_F(TestOBB, face_face ) { ASSERT_TRUE( faceFace()); }
TEST_F(TestOBB, face_edge ) { ASSERT_TRUE( faceEdge()); }
TEST_F(TestOBB, edge_edge ) { ASSERT_TRUE( edgeEdge()); }
TEST_F(TestOBB, edge_vertex ) { ASSERT_TRUE( edgeVertex()); }

TEST_F(TestCapOBB, face_vertex ) { ASSERT_TRUE( faceVertex()); }
TEST_F(TestCapOBB, face_edge ) { ASSERT_TRUE( faceEdge()); }
TEST_F(TestCapOBB, edge_vertex ) { ASSERT_TRUE( edgeVertex()); }
TEST_F(TestCapOBB, edge_edge ) { ASSERT_TRUE( edgeEdge()); }
TEST_F(TestCapOBB, vertex_edge) { ASSERT_TRUE( vertexEdge()); }
TEST_F(TestCapOBB, vertex_vertex) { ASSERT_TRUE( vertexVertex()); }

TEST_F(TestSphereOBB, vertex_sphere ) { ASSERT_TRUE( vertex()); }
TEST_F(TestSphereOBB, edge_sphere ) { ASSERT_TRUE( edge()); }
TEST_F(TestSphereOBB, face_sphere ) { ASSERT_TRUE( face()); }

TEST_F(TestTriOBB, face_face ) { ASSERT_TRUE( faceFace()); }
TEST_F(TestTriOBB, face_edge ) { ASSERT_TRUE( faceEdge()); }
TEST_F(TestTriOBB, face_vertex ) { ASSERT_TRUE( faceVertex()); }
TEST_F(TestTriOBB, edge_face ) { ASSERT_TRUE( edgeFace()); }
TEST_F(TestTriOBB, edge_edge ) { ASSERT_TRUE( edgeEdge()); }
TEST_F(TestTriOBB, edge_edge_2 ) { ASSERT_TRUE( edgeEdge2()); }
TEST_F(TestTriOBB, edge_vertex ) { ASSERT_TRUE( edgeVertex()); }
TEST_F(TestTriOBB, vertex_face ) { ASSERT_TRUE( vertexFace()); }
TEST_F(TestTriOBB, vertex_edge ) { ASSERT_TRUE( vertexEdge()); }
TEST_F(TestTriOBB, vertex_vertex ) { ASSERT_TRUE( vertexVertex()); }
