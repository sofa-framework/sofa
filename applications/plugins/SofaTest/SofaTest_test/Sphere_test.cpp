#include "stdafx.h"
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
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaComponentMain/init.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaMiscMapping/DistanceMapping.h>
#include <SofaMiscMapping/DistanceFromTargetMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseCollision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include <plugins/SceneCreator/SceneCreator.h>


#include <sofa/simulation/common/Simulation.h>
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaMeshCollision/MeshIntTool.h>
#include <SofaMeshCollision/MeshMinProximityIntersection.h>
#include <SofaMeshCollision/MeshNewProximityIntersection.inl>

#include <PrimitiveCreation.h>
#include "Sofa_test.h"

using namespace sofa::core::objectmodel;

namespace sofa {
namespace {

using namespace PrimitiveCreationTest;

struct TestSphere : public Sofa_test<double>{
    typedef sofa::defaulttype::Vec3d Vec3d;

//    /**
//      *\brief Rotates around x axis vectors x,y and z which here is a frame.
//      */
//    static void rotx(double ax,Vec3d & x,Vec3d & y,Vec3d & z);
//    static void roty(double ay,Vec3d & x,Vec3d & y,Vec3d & z);
//    static void rotz(double ay,Vec3d & x,Vec3d & y,Vec3d & z);


    bool rigidRigid1();
    bool rigidRigid2();
    bool rigidSoft1();
    bool rigidSoft2();
    bool rigidSoft3();
    bool rigidSoft4();
    bool softSoft1();


    template <class Intersector>
    bool rigidTriangle(Intersector & bi);

    template <class Intersector>
    bool softTriangle(Intersector & bi);
//    sofa::component::collision::OBB movingOBB;
//    sofa::component::collision::OBB staticOBB;
};


bool TestSphere::rigidRigid1(){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = 0;
    angles[1] = 0;
    angles[2] = 0;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    sofa::component::collision::RigidSphereModel::SPtr sphmodel2 = makeRigidSphere(Vec3d(0,0,-2),2,Vec3d(0,0,0),angles,order,scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph1(sphmodel1.get(),0);
    sofa::component::collision::RigidSphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

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


bool TestSphere::rigidRigid2(){
    double angles_1[3];
    int order_1[3];
    order_1[0] = 0;
    order_1[1] = 1;
    order_1[2] = 2;
    angles_1[0] = M_PI/8;
    angles_1[1] = M_PI/4;
    angles_1[2] = M_PI/3;

    double angles_2[3];
    int order_2[3];
    order_2[0] = 0;
    order_2[1] = 1;
    order_2[2] = 2;
    angles_2[0] = 0;
    angles_2[1] = 0;
    angles_2[2] = 0;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles_1,order_1,scn);
    sofa::component::collision::RigidSphereModel::SPtr sphmodel2 = makeRigidSphere(Vec3d(0,0,-2),2,Vec3d(0,0,0),angles_2,order_2,scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph1(sphmodel1.get(),0);
    sofa::component::collision::RigidSphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

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


bool TestSphere::rigidSoft2(){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = M_PI/8;
    angles[1] = M_PI/4;
    angles[2] = M_PI/3;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    sofa::component::collision::SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph1(sphmodel1.get(),0);
    sofa::component::collision::Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,-2)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestSphere::rigidSoft1(){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = 0;
    angles[1] = 0;
    angles[2] = 0;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    sofa::component::collision::SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph1(sphmodel1.get(),0);
    sofa::component::collision::Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,-2)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}



bool TestSphere::rigidSoft3(){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = M_PI/8;
    angles[1] = M_PI/4;
    angles[2] = M_PI/3;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    sofa::component::collision::SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph1(sphmodel1.get(),0);
    sofa::component::collision::Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph2,sph1,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,-2)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestSphere::rigidSoft4(){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = 0;
    angles[1] = 0;
    angles[2] = 0;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    sofa::component::collision::SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph1(sphmodel1.get(),0);
    sofa::component::collision::Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph2,sph1,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,-2)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

template <class Intersector>
bool TestSphere::rigidTriangle(Intersector &bi){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = 0;
    angles[1] = 0;
    angles[2] = 0;

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::RigidSphereModel::SPtr sphmodel = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-1,-1,0),Vec3d(1,-1,0),Vec3d(0,1,0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::RigidSphere sph(sphmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!bi.computeIntersection(tri,sph,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6 || detectionOUTPUT[0].normal * Vec3d(0,0,1) < 0)
        return false;

    return true;
}


template <class Intersector>
bool TestSphere::softTriangle(Intersector &bi){
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::SphereModel::SPtr sphmodel = makeSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),scn);
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3d(-1,-1,0),Vec3d(1,-1,0),Vec3d(0,1,0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::Sphere sph(sphmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!bi.computeIntersection(tri,sph,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,2.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6 || detectionOUTPUT[0].normal * Vec3d(0,0,1) < 0)
        return false;

    return true;
}


bool TestSphere::softSoft1(){
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    sofa::component::collision::SphereModel::SPtr sphmodel1 = makeSphere(Vec3d(0,0,2 + 0.01),(SReal)(2.0),Vec3d(0,0,-10),scn);
    sofa::component::collision::SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::Sphere sph1(sphmodel1.get(),0);
    sofa::component::collision::Sphere sph2(sphmodel2.get(),0);


    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

//    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
//    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3d(0,0,2.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3d(0,0,-2)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3d(0,0,1))).norm() > 1e-6|| detectionOUTPUT[0].normal * Vec3d(0,0,1) > 0)
        return false;

    return true;
}

component::collision::MinProximityIntersection::SPtr minProx = New<component::collision::MinProximityIntersection>();
component::collision::MeshMinProximityIntersection meshMin(minProx.get());

component::collision::NewProximityIntersection::SPtr newProx = New<component::collision::NewProximityIntersection>();
component::collision::MeshNewProximityIntersection meshNew(newProx.get());

TEST_F(TestSphere, rigid_rigid_1 ) { ASSERT_TRUE( rigidRigid1()); }
TEST_F(TestSphere, rigid_rigid_2 ) { ASSERT_TRUE( rigidRigid2()); }
TEST_F(TestSphere, rigid_soft_1 )  { ASSERT_TRUE( rigidSoft1()); }
TEST_F(TestSphere, rigid_soft_2 )  { ASSERT_TRUE( rigidSoft2()); }
TEST_F(TestSphere, rigid_soft_3 )  { ASSERT_TRUE( rigidSoft3()); }
TEST_F(TestSphere, rigid_soft_4 )  { ASSERT_TRUE( rigidSoft4()); }
TEST_F(TestSphere, soft_soft_1 )  { ASSERT_TRUE( softSoft1()); }
TEST_F(TestSphere, rigid_sphere_triangle_min_prox)  {ASSERT_TRUE(rigidTriangle<component::collision::MeshMinProximityIntersection >(meshMin));  }
TEST_F(TestSphere, rigid_sphere_triangle_new_prox)  {ASSERT_TRUE(rigidTriangle<component::collision::MeshNewProximityIntersection >(meshNew));  }
TEST_F(TestSphere, soft_sphere_triangle_min_prox)  {ASSERT_TRUE(softTriangle<component::collision::MeshMinProximityIntersection >(meshMin));  }
TEST_F(TestSphere, soft_sphere_triangle_new_prox)  {ASSERT_TRUE(softTriangle<component::collision::MeshNewProximityIntersection >(meshNew));  }

} // namespace
} // namespace sofa
