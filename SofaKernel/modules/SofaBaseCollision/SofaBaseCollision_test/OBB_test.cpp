/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/PrimitiveCreation.h>
#include <SofaTest/TestMessageHandler.h>

#include <SofaBaseCollision/OBBIntTool.h>
#include <SofaBaseCollision/CapsuleIntTool.h>


using namespace sofa::PrimitiveCreationTest;
using namespace sofa::defaulttype;

using sofa::core::objectmodel::New;

namespace sofa{
struct TestOBB : public Sofa_test<>{
    bool faceVertex();
    bool vertexVertex();
    bool faceFace();
    bool faceEdge();
    bool edgeEdge();
    bool edgeVertex();
};


struct TestCapOBB  : public ::testing::Test{
    bool faceVertex();
    bool faceEdge();
    bool edgeVertex();
    bool edgeEdge();
    bool vertexVertex();
    bool vertexEdge();
};

struct TestSphereOBB : public ::testing::Test{
    sofa::component::collision::RigidSphereModel::SPtr makeMyRSphere(const Vec3 & center,double radius,const Vec3 & v,
                                                                       sofa::simulation::Node::SPtr & father);

    bool vertex();
    bool edge();
    bool face();
};


struct TestTriOBB : public ::testing::Test{
    bool faceVertex();
    bool faceVertex_out();
    bool faceVertex_out2();
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

typedef sofa::component::container::MechanicalObject<sofa::defaulttype::StdRigidTypes<3, double> > MechanicalObjectRigid3d;
typedef MechanicalObjectRigid3d MechanicalObjectRigid3;

sofa::component::collision::RigidSphereModel::SPtr TestSphereOBB::makeMyRSphere(const Vec3 & center,double radius,const Vec3 & v,
                                                                   sofa::simulation::Node::SPtr & father){
    //creating node containing SphereModel
    sofa::simulation::Node::SPtr sph = father->createChild("cap");

    //creating a mechanical object which will be attached to the SphereModel
    MechanicalObjectRigid3::SPtr sphDOF = New<MechanicalObjectRigid3>();

    //editing DOF related to the SphereModel to be created, size is 1 because it contains just one Sphere
    sphDOF->resize(1);
    Data<MechanicalObjectRigid3::VecCoord> & dpositions = *sphDOF->write( sofa::core::VecId::position() );
    MechanicalObjectRigid3::VecCoord & positions = *dpositions.beginEdit();

    positions[0] = Rigid3Types::Coord(center,Quaternion(0,0,0,1));

    dpositions.endEdit();

    //Editting the velocity of the Sphere
    Data<MechanicalObjectRigid3::VecDeriv> & dvelocities = *sphDOF->write( sofa::core::VecId::velocity() );

    MechanicalObjectRigid3::VecDeriv & velocities = *dvelocities.beginEdit();
    velocities[0] = v;
    dvelocities.endEdit();

    sph->addObject(sphDOF);

    //creating an OBBModel and attaching it to the same node than obbDOF
    sofa::component::collision::RigidSphereModel::SPtr sphCollisionModel = New<sofa::component::collision::RigidSphereModel >();
    sph->addObject(sphCollisionModel);


    //editting the OBBModel
    sphCollisionModel->init();
    Data<sofa::component::collision::RigidSphereModel::VecReal> & dVecReal = sphCollisionModel->radius;
    sofa::component::collision::RigidSphereModel::VecReal & vecReal = *(dVecReal.beginEdit());

    vecReal[0] = radius;

    dVecReal.endEdit();

    return sphCollisionModel;
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
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //the second OBB which is moving, one OBB must move, if not, there is no collision (OBB collision algorithm is like that)
    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3(0,0,sqrt(3.0) + 0.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb1 (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    //in the other sens//////////////////////

    detectionOUTPUT.clear();

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb1,obb0,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of obb0 (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb1 (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3(0,0,sqrt(3.0) + 0.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT)){
        return false;
    }

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //in the other sens//////////////////////

    detectionOUTPUT.clear();

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb1,obb0,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    return true;
}

//obb0's face 3,2,6,7 in intersection with obb1's face 0,4,5,1 (see indexation above)
bool TestOBB::faceFace(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3(0,1,1.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0.5,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0.5,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    //in the other sens//////////////////////

    detectionOUTPUT.clear();

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb1,obb0,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of obb0 (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0.5,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb1 (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3(0,0.5,0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

//obb0's face 3,2,6,7 in intersection with obb1's edge 3-0
bool TestOBB::faceEdge(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = M_PI_2;
    angles[2] = M_PI_4;
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3(0,0,sqrt(2.0) + 0.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    //in the other sens//////////////////////

    detectionOUTPUT.clear();

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb1,obb0,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of obb0 (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the intersection point of obb1 (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3(0,0,sqrt(2.0) + 0.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(-1,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(-1,0,0.01)).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel0 = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    order[0] = 2;
    order[1] = 1;
    order[2] = 0;
    angles[0] = 0;
    angles[1] = acos(1/sqrt(3.0));
    angles[2] = M_PI_4;
    sofa::component::collision::OBBModel::SPtr obbmodel1 = makeOBB(Vec3(0,0,sqrt(3.0) + 0.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

    sofa::component::collision::OBB obb0(obbmodel0.get(),0);
    sofa::component::collision::OBB obb1(obbmodel1.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::OBBIntTool::computeIntersection(obb0,obb1,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    return true;
}

bool TestCapOBB::faceVertex(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel =
            makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3(0,0,1 + 0.01),Vec3(0,0,2),1,Vec3(0,0,-10),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

bool TestCapOBB::faceEdge(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel =
            makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3(-1,0,1 + 0.01),Vec3(1,0,1 + 0.01),1,Vec3(0,0,-10),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
            makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3(0,0,1 + 0.01),Vec3(0,0,2),1,Vec3(0,0,-10),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
            makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3(-0.5,0,1 + 0.01),Vec3(0.5,0,1 + 0.01),1,Vec3(0,0,-10),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3(-0.5,0,1 + 0.01),Vec3(0.5,0,1 + 0.01),1,Vec3(0,0,-10),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::CapsuleModel::SPtr capmodel = makeCap(Vec3(0,0,1 + 0.01),Vec3(0,0,2),1,Vec3(0,0,-10),scn);

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
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::RigidSphereModel::SPtr sphmodel = makeMyRSphere(Vec3(0,0,1 + 0.01),1,Vec3(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::RigidSphere sph(sphmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(sph,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::RigidSphereModel::SPtr sphmodel = makeMyRSphere(Vec3(0,0,1 + 0.01),1,Vec3(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::RigidSphere sph(sphmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(sph,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestSphereOBB::face(){
    //first, we create the transformation to make the first OBB (which is axes aligned)
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};

    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);//this OBB is not moving and the contact face will be z = 0 since
                                        //the center of this OBB is (0,0,-1) and its extent is 1

    //we construct the falling capsule
    sofa::component::collision::RigidSphereModel::SPtr sphmodel = makeMyRSphere(Vec3(0,0,1 + 0.01),1,Vec3(0,0,-10),scn);

    //we construct the OBB and the capsule from the OBBModel and the CapsuleModel
    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::RigidSphere sph(sphmodel.get(),0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if(!sofa::component::collision::OBBIntTool::computeIntersection(sph,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}

bool TestTriOBB::faceFace(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(-2,-2,0.01),Vec3(-2,2,0.01),Vec3(2,0,0.01),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::faceVertex_out(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(-1.01,0,1.01),angles,order,Vec3(0,0,-10),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,0,0),Vec3(2,2,0),Vec3(2,-2,0),Vec3(0,0,0),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //triangle point
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //obb point
    if((detectionOUTPUT[0].point[1] - Vec3(-0.01,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::faceVertex_out2(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(-1.01,0,-1.01),angles,order,Vec3(0,0,10),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,0,0),Vec3(2,2,0),Vec3(2,-2,0),Vec3(0,0,0),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    std::cout<<"detectionOUTPUT[0].point[0] "<<detectionOUTPUT[0].point[0]<<std::endl;
    std::cout<<"detectionOUTPUT[0].point[1] "<<detectionOUTPUT[0].point[1]<<std::endl;

    //triangle point
    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    //obb point
    if((detectionOUTPUT[0].point[1] - Vec3(-0.01,0,-0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::faceEdge(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,-2,0.01),Vec3(0,2,0.01),Vec3(2,0,2),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
        return false;

    return true;
}


bool TestTriOBB::faceVertex(){
    double angles[3] = {0,0,0};
    int order[3] = {0,1,2};
    sofa::simulation::Node::SPtr scn = New<sofa::simulation::tree::GNode>();
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-1),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,-2,2),Vec3(0,2,2),Vec3(0,0,0.01),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(-2,-2,0.01),Vec3(-2,2,0.01),Vec3(2,0,0.01),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,-2,0.01),Vec3(0,2,0.01),Vec3(2,0,2),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(-1,0,0.01),Vec3(1,0,0.01),Vec3(2,0,2),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(2.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,0,0.01),Vec3(1,0,2),Vec3(-1,0,2),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(-2,-2,0.01),Vec3(-2,2,0.01),Vec3(2,0,0.01),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(-1,0,0.01),Vec3(1,0,0.01),Vec3(2,0,2),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(1,0,0.01)).norm() > 1e-6 && (detectionOUTPUT[0].point[0] - Vec3(-1,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(1,0,0)).norm() > 1e-6 && (detectionOUTPUT[0].point[1] - Vec3(-1,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
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
    sofa::component::collision::OBBModel::SPtr obbmodel = makeOBB(Vec3(0,0,-sqrt(3.0)),angles,order,Vec3(0,0,0),Vec3(1,1,1),scn);

    int tri_flg = sofa::component::collision::TriangleModel::FLAG_POINTS | sofa::component::collision::TriangleModel::FLAG_EDGES;
    sofa::component::collision::TriangleModel::SPtr trimodel = makeTri(Vec3(0,0,0.01),Vec3(1,0,2),Vec3(-1,0,2),Vec3(0,0,-10),scn);

    sofa::component::collision::OBB obb(obbmodel.get(),0);
    sofa::component::collision::Triangle tri(trimodel.get(),0);

    sofa::helper::vector<sofa::core::collision::DetectionOutput> detectionOUTPUT;

    if(!sofa::component::collision::MeshIntTool::computeIntersection(tri,tri_flg,obb,1.0,1.0,&detectionOUTPUT))
        return false;

    if((detectionOUTPUT[0].point[0] - Vec3(0,0,0.01)).norm() > 1e-6)
        return false;

    if((detectionOUTPUT[0].point[1] - Vec3(0,0,0)).norm() > 1e-6)
        return false;

//    if((detectionOUTPUT[0].normal.cross(Vec3(0,0,1))).norm() > 1e-6)
//        return false;

    return true;
}

TEST_F(TestOBB, face_vertex ) {
    ASSERT_TRUE( faceVertex());
}

TEST_F(TestOBB, vertex_vertex ) {
    ASSERT_TRUE( vertexVertex());
}

TEST_F(TestOBB, face_face ) {
    ASSERT_TRUE( faceFace());
}

TEST_F(TestOBB, face_edge ) {
    ASSERT_TRUE( faceEdge());
}

TEST_F(TestOBB, edge_edge ) {
    ASSERT_TRUE( edgeEdge());
}

TEST_F(TestOBB, edge_vertex ) {
    ASSERT_TRUE( edgeVertex());
}

TEST_F(TestCapOBB, face_vertex ) {
    ASSERT_TRUE( faceVertex());
}

TEST_F(TestCapOBB, face_edge ) {
    ASSERT_TRUE( faceEdge());
}

TEST_F(TestCapOBB, edge_vertex ) {
    ASSERT_TRUE( edgeVertex());
}

TEST_F(TestCapOBB, edge_edge ) {
    ASSERT_TRUE( edgeEdge());
}

TEST_F(TestCapOBB, vertex_edge) {
    ASSERT_TRUE( vertexEdge());
}

TEST_F(TestCapOBB, vertex_vertex) {
    ASSERT_TRUE( vertexVertex());
}

TEST_F(TestSphereOBB, vertex_sphere ) {
    ASSERT_TRUE( vertex());
}

TEST_F(TestSphereOBB, edge_sphere ) {
    ASSERT_TRUE( edge());
}

TEST_F(TestSphereOBB, face_sphere ) {
    ASSERT_TRUE( face());
}

TEST_F(TestTriOBB, face_face ) {
    ASSERT_TRUE( faceFace());
}

TEST_F(TestTriOBB, face_edge ) {
    ASSERT_TRUE( faceEdge());
}

TEST_F(TestTriOBB, face_vertex ) {
    ASSERT_TRUE( faceVertex());
}

TEST_F(TestTriOBB, edge_face ) {
    ASSERT_TRUE( edgeFace());
}

TEST_F(TestTriOBB, edge_edge ) {
    ASSERT_TRUE( edgeEdge());
}

TEST_F(TestTriOBB, edge_edge_2 ) {
    ASSERT_TRUE( edgeEdge2());
}

TEST_F(TestTriOBB, edge_vertex ) {
    ASSERT_TRUE( edgeVertex());
}

TEST_F(TestTriOBB, vertex_face ) {
    ASSERT_TRUE( vertexFace());
}

TEST_F(TestTriOBB, vertex_edge ) {
    ASSERT_TRUE( vertexEdge());
}

TEST_F(TestTriOBB, vertex_vertex ) {
    ASSERT_TRUE( vertexVertex());
}

TEST_F(TestTriOBB, face_vertex_out ) {
    ASSERT_TRUE( faceVertex_out());
}

TEST_F(TestTriOBB, face_vertex_out2 ) {
    ASSERT_TRUE( faceVertex_out2());
}

}
