/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/config.h>
#include <cmath>

#include <vector>
using std::vector;

#include <string>
using std::string;

#include<sofa/core/objectmodel/BaseObject.h>
using sofa::core::objectmodel::BaseObject ;

#include<sofa/simulation/Node.h>
using sofa::simulation::Node ;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <SofaTest/PrimitiveCreation.h>

#include <SofaGeneralMeshCollision/MeshMinProximityIntersection.h>
using sofa::component::collision::MeshMinProximityIntersection;

#include <SofaMeshCollision/MeshNewProximityIntersection.inl>
using sofa::component::collision::MeshNewProximityIntersection ;

using sofa::core::ExecParams ;
using sofa::core::objectmodel::New;
using sofa::component::collision::Sphere;
using sofa::component::collision::SphereModel ;
using sofa::component::collision::TriangleModel;
using sofa::component::collision::RigidSphereModel;
using sofa::component::collision::RigidSphere;
using sofa::component::collision::BaseIntTool;
using sofa::core::collision::DetectionOutput;
using sofa::defaulttype::Vec3d;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <SofaTest/TestMessageHandler.h>

namespace sofa {

using namespace PrimitiveCreationTest;

struct TestSphere : public Sofa_test<>{
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

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    RigidSphereModel::SPtr sphmodel2 = makeRigidSphere(Vec3d(0,0,-2),2,Vec3d(0,0,0),angles,order,scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    RigidSphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
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

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles_1,order_1,scn);
    RigidSphereModel::SPtr sphmodel2 = makeRigidSphere(Vec3d(0,0,-2),2,Vec3d(0,0,0),angles_2,order_2,scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    RigidSphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
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


bool TestSphere::rigidSoft2(){
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = M_PI/8;
    angles[1] = M_PI/4;
    angles[2] = M_PI/3;

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

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

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

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

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph2,sph1,1.0,1.0,&detectionOUTPUT))
        return false;

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

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel1 = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph2,sph1,1.0,1.0,&detectionOUTPUT))
        return false;

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

   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    RigidSphereModel::SPtr sphmodel = makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    TriangleModel::SPtr trimodel = makeTri(Vec3d(-1,-1,0),Vec3d(1,-1,0),Vec3d(0,1,0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    RigidSphere sph(sphmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!bi.computeIntersection(tri,sph,&detectionOUTPUT))
        return false;

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
   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    SphereModel::SPtr sphmodel = makeSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),scn);
    TriangleModel::SPtr trimodel = makeTri(Vec3d(-1,-1,0),Vec3d(1,-1,0),Vec3d(0,1,0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    Sphere sph(sphmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!bi.computeIntersection(tri,sph,&detectionOUTPUT))
        return false;

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
   Node::SPtr scn = New<sofa::simulation::tree::GNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    SphereModel::SPtr sphmodel1 = makeSphere(Vec3d(0,0,2 + 0.01),(SReal)(2.0),Vec3d(0,0,-10),scn);
    SphereModel::SPtr sphmodel2 = makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    Sphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);


    sofa::helper::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!BaseIntTool::computeIntersection(sph1,sph2,1.0,1.0,&detectionOUTPUT))
        return false;

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


void checkAttributes()
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Vec3d'/>                                         \n"
             "   <SphereModel name='spheremodel'/>                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* theSphere = root->getTreeNode("Level 1")->getObject("spheremodel") ;
    EXPECT_NE(theSphere, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    vector<string> attrnames = {
        "listRadius", "radius"
    };

    for(auto& attrname : attrnames)
        EXPECT_NE( theSphere->findData(attrname), nullptr ) << "Missing attribute with name '" << attrname << "'." ;
}

void checkSceneWithVec3MechanicalModel()
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Vec3d'/>                                         \n"
             "   <SphereModel name='spheremodel'/>                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* theSphere = root->getTreeNode("Level 1")->getObject("spheremodel") ;
    EXPECT_NE(theSphere, nullptr) ;
}

void checkSceneWithRigid3dMechanicalModel()
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Rigid3d'/>                                       \n"
             "   <SphereModel name='spheremodel'/>                                           \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

    BaseObject* theSphere = root->getTreeNode("Level 1")->getObject("spheremodel") ;
    EXPECT_NE(theSphere, nullptr) ;
}

void checkGracefulHandlingWhenMechanicalModelIsMissing()
{
    EXPECT_MSG_EMIT(Error) ;

    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <SphereModel name='spheremodel' template='Vec3d'/>                          \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    Node::SPtr root = SceneLoaderXML::loadFromMemory ("testscene",
                                                      scene.str().c_str(),
                                                      scene.str().size()) ;
    EXPECT_NE(root.get(), nullptr) ;
    root->init(ExecParams::defaultInstance()) ;

}

component::collision::MinProximityIntersection::SPtr minProx = New<component::collision::MinProximityIntersection>();
MeshMinProximityIntersection meshMin(minProx.get());

component::collision::NewProximityIntersection::SPtr newProx = New<component::collision::NewProximityIntersection>();
MeshNewProximityIntersection meshNew(newProx.get());

TEST_F(TestSphere, rigid_rigid_1 ) { ASSERT_TRUE( rigidRigid1()); }
TEST_F(TestSphere, rigid_rigid_2 ) { ASSERT_TRUE( rigidRigid2()); }
TEST_F(TestSphere, rigid_soft_1 )  { ASSERT_TRUE( rigidSoft1()); }
TEST_F(TestSphere, rigid_soft_2 )  { ASSERT_TRUE( rigidSoft2()); }
TEST_F(TestSphere, rigid_soft_3 )  { ASSERT_TRUE( rigidSoft3()); }
TEST_F(TestSphere, rigid_soft_4 )  { ASSERT_TRUE( rigidSoft4()); }
TEST_F(TestSphere, soft_soft_1 )  { ASSERT_TRUE( softSoft1()); }
TEST_F(TestSphere, rigid_sphere_triangle_min_prox)  {ASSERT_TRUE(rigidTriangle<MeshMinProximityIntersection >(meshMin));  }
TEST_F(TestSphere, rigid_sphere_triangle_new_prox)  {ASSERT_TRUE(rigidTriangle<MeshNewProximityIntersection >(meshNew));  }
TEST_F(TestSphere, soft_sphere_triangle_min_prox)  {ASSERT_TRUE(softTriangle<MeshMinProximityIntersection >(meshMin));  }
TEST_F(TestSphere, soft_sphere_triangle_new_prox)  {ASSERT_TRUE(softTriangle<MeshNewProximityIntersection >(meshNew));  }


TEST_F(TestSphere, checkSceneWithVec3MechanicalModel)
{
    checkSceneWithVec3MechanicalModel();
}

TEST_F(TestSphere, checkSceneWithRigid3dMechanicalMode)
{
   checkSceneWithRigid3dMechanicalModel();
}

TEST_F(TestSphere, checkAttributes)
{
    checkAttributes();
}

TEST_F(TestSphere, checkGracefulHandlingWhenMechanicalModelIsMissing)
{
    checkGracefulHandlingWhenMechanicalModelIsMissing();
}


}
