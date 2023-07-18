/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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

#include <sofa/simulation/common/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML ;

#include <sofa/component/collision/detection/intersection/MinProximityIntersection.h>
using sofa::component::collision::detection::intersection::MinProximityIntersection;

using sofa::core::execparams::defaultInstance; 
using sofa::core::objectmodel::New;

#include <sofa/component/collision/geometry/SphereModel.h>
using sofa::component::collision::geometry::Sphere;
using sofa::component::collision::geometry::SphereCollisionModel ;
using sofa::component::collision::geometry::RigidSphere;

using sofa::core::collision::DetectionOutput;
using sofa::type::Vec3d;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher ;

#include <sofa/helper/logging/ClangMessageHandler.h>
using sofa::helper::logging::ClangMessageHandler ;

#include <sofa/simulation/graph/DAGNode.h>

#include <sofa/testing/BaseSimulationTest.h>
using sofa::testing::BaseSimulationTest;

#include <sofa/component/collision/testing/SpherePrimitiveCreator.h>

namespace sofa {

struct TestSphere : public BaseSimulationTest
{
    void SetUp() override
    {
        m_proxIntersection = sofa::core::objectmodel::New<MinProximityIntersection>();
        m_proxIntersection->setAlarmDistance(1.0);
        m_proxIntersection->setContactDistance(1.0);
    }
    void TearDown() override
    {
    }

    bool rigidRigid1();
    bool rigidRigid2();
    bool rigidSoft1();
    bool rigidSoft2();
    bool rigidSoft3();
    bool rigidSoft4();
    bool softSoft1();

    MinProximityIntersection::SPtr m_proxIntersection;
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

    Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
    //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel1 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel2 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,-2),2,Vec3d(0,0,0),angles,order,scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    RigidSphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //looking for an intersection (with proximities)
    if(!m_proxIntersection->computeIntersection(sph1,sph2,&detectionOUTPUT))
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

   Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel1 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles_1,order_1,scn);
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel2 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,-2),2,Vec3d(0,0,0),angles_2,order_2,scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    RigidSphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!m_proxIntersection->computeIntersection(sph1,sph2,&detectionOUTPUT))
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

   Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel1 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel2 = sofa::collision_test::makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!m_proxIntersection->computeIntersection(sph1,sph2,&detectionOUTPUT))
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

   Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel1 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel2 = sofa::collision_test::makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!m_proxIntersection->computeIntersection(sph1,sph2,&detectionOUTPUT))
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

   Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel1 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel2 = sofa::collision_test::makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!m_proxIntersection->computeIntersection(sph2,sph1,&detectionOUTPUT))
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

   Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel1 = sofa::collision_test::makeRigidSphere(Vec3d(0,0,2 + 0.01),2,Vec3d(0,0,-10),angles,order,scn);
    const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel2 = sofa::collision_test::makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!m_proxIntersection->computeIntersection(sph2,sph1,&detectionOUTPUT))
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


bool TestSphere::softSoft1(){
   Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling sphere
   const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel1 = sofa::collision_test::makeSphere(Vec3d(0,0,2 + 0.01),(SReal)(2.0),Vec3d(0,0,-10),scn);
   const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel2 = sofa::collision_test::makeSphere(Vec3d(0,0,-2),(SReal)(2.0),Vec3d(0,0,0),scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    Sphere sph1(sphmodel1.get(),0);
    Sphere sph2(sphmodel2.get(),0);


    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!m_proxIntersection->computeIntersection(sph1,sph2,&detectionOUTPUT))
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
             "   <MechanicalObject template='Vec3d'/>                                        \n"
             "   <SphereCollisionModel name='spheremodel'/>                                  \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    EXPECT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* theSphere = root->getTreeNode("Level 1")->getObject("spheremodel") ;
    EXPECT_NE(theSphere, nullptr) ;

    /// List of the supported attributes the user expect to find
    /// This list needs to be updated if you add an attribute.
    const vector<string> attrnames = {
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
             "   <MechanicalObject template='Vec3d'/>                                        \n"
             "   <SphereCollisionModel name='spheremodel'/>                                  \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    EXPECT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

    BaseObject* theSphere = root->getTreeNode("Level 1")->getObject("spheremodel") ;
    EXPECT_NE(theSphere, nullptr) ;
}

void checkSceneWithRigid3dMechanicalModel()
{
    std::stringstream scene ;
    scene << "<?xml version='1.0'?>"
             "<Node 	name='Root' gravity='0 -9.81 0' time='0' animate='0' >               \n"
             "  <Node name='Level 1'>                                                        \n"
             "   <MechanicalObject template='Rigid3d'/>                                      \n"
             "   <SphereCollisionModel name='spheremodel'/>                                  \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    EXPECT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

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
             "   <SphereCollisionModel name='spheremodel' template='Vec3d'/>                 \n"
             "  </Node>                                                                      \n"
             "</Node>                                                                        \n" ;

    const Node::SPtr root = SceneLoaderXML::loadFromMemory("testscene", scene.str().c_str());
    EXPECT_NE(root.get(), nullptr) ;
    root->init(sofa::core::execparams::defaultInstance()) ;

}

TEST_F(TestSphere, rigid_rigid_1 ) { ASSERT_TRUE( rigidRigid1()); }
TEST_F(TestSphere, rigid_rigid_2 ) { ASSERT_TRUE( rigidRigid2()); }
TEST_F(TestSphere, rigid_soft_1 )  { ASSERT_TRUE( rigidSoft1()); }
TEST_F(TestSphere, rigid_soft_2 )  { ASSERT_TRUE( rigidSoft2()); }
TEST_F(TestSphere, rigid_soft_3 )  { ASSERT_TRUE( rigidSoft3()); }
TEST_F(TestSphere, rigid_soft_4 )  { ASSERT_TRUE( rigidSoft4()); }
TEST_F(TestSphere, soft_soft_1 )  { ASSERT_TRUE( softSoft1()); }


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
