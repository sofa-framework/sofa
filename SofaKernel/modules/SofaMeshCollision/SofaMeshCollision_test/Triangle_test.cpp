
#include <SofaGeneralMeshCollision/MeshMinProximityIntersection.h>
using sofa::component::collision::MeshMinProximityIntersection;

#include <SofaMeshCollision/MeshNewProximityIntersection.inl>
using sofa::component::collision::MeshNewProximityIntersection;

#include <SofaMeshCollision/TriangleModel.h>

using sofa::core::execparams::defaultInstance;
using sofa::core::objectmodel::New;
using sofa::component::collision::Sphere;
using sofa::component::collision::SphereCollisionModel;
using sofa::component::collision::RigidSphere;
using sofa::component::collision::TriangleCollisionModel;

using sofa::component::collision::TriangleCollisionModel;


using sofa::core::collision::DetectionOutput;
using sofa::type::Vec3d;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <SofaSimulationGraph/DAGNode.h>

#include "MeshPrimitiveCreator.h"
#include <SofaBaseCollision_test/SpherePrimitiveCreator.h>

namespace sofa 
{
    struct TestTriangle : public BaseTest
    {
        void SetUp() override
        {
        }
        void TearDown() override
        {

        }

        template <class Intersector>
        bool rigidTriangle(Intersector& bi);

        template <class Intersector>
        bool softTriangle(Intersector& bi);
    };

template <class Intersector>
bool TestTriangle::rigidTriangle(Intersector& bi) {
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
    SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel = sofa::collision_test::makeRigidSphere(Vec3d(0, 0, 2 + 0.01), 2, Vec3d(0, 0, -10), angles, order, scn);
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr trimodel = sofa::collision_test::makeTri(Vec3d(-1, -1, 0), Vec3d(1, -1, 0), Vec3d(0, 1, 0), Vec3d(0, 0, 0), scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph(sphmodel.get(), 0);
    sofa::component::collision::Triangle tri(trimodel.get(), 0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if (!bi.computeIntersection(tri, sph, &detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if ((detectionOUTPUT[0].point[0] - Vec3d(0, 0, 0)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if ((detectionOUTPUT[0].point[1] - Vec3d(0, 0, 0.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if ((detectionOUTPUT[0].normal.cross(Vec3d(0, 0, 1))).norm() > 1e-6 || detectionOUTPUT[0].normal * Vec3d(0, 0, 1) < 0)
        return false;

    return true;
}


template <class Intersector>
bool TestTriangle::softTriangle(Intersector& bi) {
    Node::SPtr scn = New<sofa::simulation::graph::DAGNode>();
    //the center of this OBB is (0,0,-1) and its extent is 1

//we construct the falling sphere
    SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel = sofa::collision_test::makeSphere(Vec3d(0, 0, 2 + 0.01), 2, Vec3d(0, 0, -10), scn);
    TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr trimodel = sofa::collision_test::makeTri(Vec3d(-1, -1, 0), Vec3d(1, -1, 0), Vec3d(0, 1, 0), Vec3d(0, 0, 0), scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    Sphere sph(sphmodel.get(), 0);
    sofa::component::collision::Triangle tri(trimodel.get(), 0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if (!bi.computeIntersection(tri, sph, &detectionOUTPUT))
        return false;

    //the intersection point of cap (detectionOUTPUT[0].point[1]) should be (0,0,0.01)
    if ((detectionOUTPUT[0].point[0] - Vec3d(0, 0, 0)).norm() > 1e-6)
        return false;

    //the intersection point of obb (detectionOUTPUT[0].point[0]) should be (0,0,0)
    if ((detectionOUTPUT[0].point[1] - Vec3d(0, 0, 2.01)).norm() > 1e-6)
        return false;

    //the contact response direction (detectionOUTPUT[0].normal) should be (0,0,1)
    if ((detectionOUTPUT[0].normal.cross(Vec3d(0, 0, 1))).norm() > 1e-6 || detectionOUTPUT[0].normal * Vec3d(0, 0, 1) < 0)
        return false;

    return true;
}

component::collision::MinProximityIntersection::SPtr minProx = New<component::collision::MinProximityIntersection>();
MeshMinProximityIntersection meshMin(minProx.get());

component::collision::NewProximityIntersection::SPtr newProx = New<component::collision::NewProximityIntersection>();
MeshNewProximityIntersection meshNew(newProx.get());

TEST_F(TestTriangle, rigid_sphere_triangle_min_prox) { ASSERT_TRUE(rigidTriangle<MeshMinProximityIntersection >(meshMin)); }
TEST_F(TestTriangle, rigid_sphere_triangle_new_prox) { ASSERT_TRUE(rigidTriangle<MeshNewProximityIntersection >(meshNew)); }
TEST_F(TestTriangle, soft_sphere_triangle_min_prox) { ASSERT_TRUE(softTriangle<MeshMinProximityIntersection >(meshMin)); }
TEST_F(TestTriangle, soft_sphere_triangle_new_prox) { ASSERT_TRUE(softTriangle<MeshNewProximityIntersection >(meshNew)); }

} 
