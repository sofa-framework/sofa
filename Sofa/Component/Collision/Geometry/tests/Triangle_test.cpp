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

#include <sofa/component/collision/detection/intersection/MinProximityIntersection.h>
#include <sofa/component/collision/detection/intersection/MeshMinProximityIntersection.h>
using sofa::component::collision::detection::intersection::MeshMinProximityIntersection;

#include <sofa/component/collision/detection/intersection/MeshNewProximityIntersection.inl>
using sofa::component::collision::detection::intersection::MeshNewProximityIntersection;

#include <sofa/component/collision/geometry/TriangleModel.h>

using sofa::core::execparams::defaultInstance;
using sofa::core::objectmodel::New;
#include <sofa/component/collision/geometry/SphereModel.h>
using sofa::component::collision::geometry::Sphere;
using sofa::component::collision::geometry::SphereCollisionModel;
using sofa::component::collision::geometry::RigidSphere;
using sofa::component::collision::geometry::TriangleCollisionModel;


using sofa::core::collision::DetectionOutput;
using sofa::type::Vec3d;

#include <sofa/helper/logging/Messaging.h>
using sofa::helper::logging::MessageDispatcher;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/simulation/Node.h>

#include <sofa/component/collision/testing/MeshPrimitiveCreator.h>
#include <sofa/component/collision/testing/SpherePrimitiveCreator.h>

namespace sofa 
{
    struct TestTriangle : public BaseTest
    {
        void doSetUp() override
        {
        }
        void doTearDown() override
        {

        }

        template <class Intersector>
        bool rigidTriangle(sofa::component::collision::detection::intersection::BaseProximityIntersection::SPtr intersectionMethod, Intersector& bi);

        template <class Intersector>
        bool softTriangle(sofa::component::collision::detection::intersection::BaseProximityIntersection::SPtr intersectionMethod, Intersector& bi);
    };

template <class Intersector>
bool TestTriangle::rigidTriangle(sofa::component::collision::detection::intersection::BaseProximityIntersection::SPtr intersectionMethod, Intersector& bi) {
    double angles[3];
    int order[3];
    order[0] = 0;
    order[1] = 1;
    order[2] = 2;
    angles[0] = 0;
    angles[1] = 0;
    angles[2] = 0;

    Node::SPtr scn = New<sofa::simulation::Node>();
    //the center of this OBB is (0,0,-1) and its extent is 1

//we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Rigid3Types>::SPtr sphmodel = sofa::collision_test::makeRigidSphere(Vec3d(0, 0, 2 + 0.01), 2, Vec3d(0, 0, -10), angles, order, scn);
    const TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr trimodel = sofa::collision_test::makeTri(Vec3d(-1, -1, 0), Vec3d(1, -1, 0), Vec3d(0, 1, 0), Vec3d(0, 0, 0), scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    RigidSphere sph(sphmodel.get(), 0);
    sofa::component::collision::geometry::Triangle tri(trimodel.get(), 0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if (!bi.computeIntersection(tri, sph, &detectionOUTPUT, intersectionMethod.get()))
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
bool TestTriangle::softTriangle(sofa::component::collision::detection::intersection::BaseProximityIntersection::SPtr intersectionMethod, Intersector& bi) {
    Node::SPtr scn = New<sofa::simulation::Node>();
    //the center of this OBB is (0,0,-1) and its extent is 1

//we construct the falling sphere
    const SphereCollisionModel<sofa::defaulttype::Vec3Types>::SPtr sphmodel = sofa::collision_test::makeSphere(Vec3d(0, 0, 2 + 0.01), 2, Vec3d(0, 0, -10), scn);
    const TriangleCollisionModel<sofa::defaulttype::Vec3Types>::SPtr trimodel = sofa::collision_test::makeTri(Vec3d(-1, -1, 0), Vec3d(1, -1, 0), Vec3d(0, 1, 0), Vec3d(0, 0, 0), scn);


    //we construct the OBB and the capsule from the OBBCollisionModel<sofa::defaulttype::Rigid3Types> and the CapsuleModel
    Sphere sph(sphmodel.get(), 0);
    sofa::component::collision::geometry::Triangle tri(trimodel.get(), 0);

    //collision configuration is such that the face defined by 3,2,6,7 vertices of obb0 (not moving) is intersected
    //at its center by the vertex 0 of obb1 (moving)

    sofa::type::vector<DetectionOutput> detectionOUTPUT;

    //loooking for an intersection
    if (!bi.computeIntersection(tri, sph, &detectionOUTPUT, intersectionMethod.get()))
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

component::collision::detection::intersection::MinProximityIntersection::SPtr minProx = New<component::collision::detection::intersection::MinProximityIntersection>();
MeshMinProximityIntersection meshMin(minProx.get());

component::collision::detection::intersection::NewProximityIntersection::SPtr newProx = New<component::collision::detection::intersection::NewProximityIntersection>();
MeshNewProximityIntersection meshNew(newProx.get());

TEST_F(TestTriangle, rigid_sphere_triangle_min_prox) { ASSERT_TRUE(rigidTriangle<MeshMinProximityIntersection >(minProx, meshMin)); }
TEST_F(TestTriangle, rigid_sphere_triangle_new_prox) { ASSERT_TRUE(rigidTriangle<MeshNewProximityIntersection >(newProx, meshNew)); }
TEST_F(TestTriangle, soft_sphere_triangle_min_prox) { ASSERT_TRUE(softTriangle<MeshMinProximityIntersection >(minProx, meshMin)); }
TEST_F(TestTriangle, soft_sphere_triangle_new_prox) { ASSERT_TRUE(softTriangle<MeshNewProximityIntersection >(newProx, meshNew)); }

} 
