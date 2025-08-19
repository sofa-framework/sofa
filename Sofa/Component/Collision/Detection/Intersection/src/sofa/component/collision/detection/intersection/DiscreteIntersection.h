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
#pragma once
#include <sofa/component/collision/detection/intersection/config.h>

#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/IntersectorFactory.h>
#include <sofa/component/collision/geometry/SphereModel.h>
#include <sofa/component/collision/geometry/CubeModel.h>

namespace sofa::component::collision::detection::intersection
{
class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API DiscreteIntersection : public core::collision::Intersection, public core::collision::BaseIntersector
{
public:
    SOFA_CLASS(DiscreteIntersection,sofa::core::collision::Intersection);
protected:
    DiscreteIntersection();
	~DiscreteIntersection() override { }
	
public:
    /// Return the intersector class handling the given pair of collision models, or nullptr if not supported.
    /// @param swapModel output value set to true if the collision models must be swapped before calling the intersector.
    core::collision::ElementIntersector* findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels) override;

    core::collision::IntersectorMap intersectors;
    typedef core::collision::IntersectorFactory<DiscreteIntersection> IntersectorFactory;

    //Intersectors
    // Cube
    virtual bool testIntersection(collision::geometry::Cube& cube1, collision::geometry::Cube& cube2, const core::collision::Intersection* currentIntersection);
    virtual int computeIntersection(collision::geometry::Cube& cube1, collision::geometry::Cube& cube2, OutputVector* contacts, const core::collision::Intersection* currentIntersection);

    //Sphere
    virtual bool testIntersection(collision::geometry::Sphere& sph1, collision::geometry::Sphere& sph2, const core::collision::Intersection* currentIntersection);
    virtual int computeIntersection(collision::geometry::Sphere& sph1, collision::geometry::Sphere& sph2, OutputVector* contacts, const core::collision::Intersection* currentIntersection);
    virtual bool testIntersection(collision::geometry::RigidSphere& sph1, collision::geometry::RigidSphere& sph2, const core::collision::Intersection* currentIntersection);
    virtual int computeIntersection(collision::geometry::RigidSphere& sph1, collision::geometry::RigidSphere& sph2, OutputVector* contacts, const core::collision::Intersection* currentIntersection);
    virtual bool testIntersection(collision::geometry::Sphere& sph1, collision::geometry::RigidSphere& sph2, const core::collision::Intersection* currentIntersection);
    virtual int computeIntersection(collision::geometry::Sphere& sph1, collision::geometry::RigidSphere& sph2, OutputVector* contacts, const core::collision::Intersection* currentIntersection);

protected:

    template<class SphereType1, class SphereType2>
    bool testIntersectionSphere(SphereType1& sph1, SphereType2& sph2, const SReal alarmDist)
    {
        const auto r = sph1.r() + sph2.r() + alarmDist;
        return (sph1.center() - sph2.center()).norm2() <= r * r;
    }

    template<class SphereType1, class SphereType2>
    int computeIntersectionSphere(SphereType1& sph1, SphereType2& sph2, DiscreteIntersection::OutputVector* contacts, const SReal alarmDist, const SReal contactDist)
    {
        const SReal r = sph1.r() + sph2.r();
        const SReal myAlarmDist = alarmDist + r;
        const type::Vec3 dist = sph2.center() - sph1.center();
        const SReal norm2 = dist.norm2();

        if (norm2 > myAlarmDist * myAlarmDist)
            return 0;

        contacts->resize(contacts->size() + 1);
        core::collision::DetectionOutput* detection = &*(contacts->end() - 1);
        const SReal distSph1Sph2 = helper::rsqrt(norm2);
        detection->normal = dist / distSph1Sph2;
        detection->point[0] = sph1.getContactPointByNormal(-detection->normal);
        detection->point[1] = sph2.getContactPointByNormal(detection->normal);

        detection->value = distSph1Sph2 - r - contactDist;
        detection->elem.first = sph1;
        detection->elem.second = sph2;
        detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

        return 1;
    }
};

} // namespace sofa::component::collision::detection::intersection

namespace sofa::core::collision
{
#if !defined(SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_CPP)
extern template class SOFA_COMPONENT_COLLISION_DETECTION_INTERSECTION_API IntersectorFactory<component::collision::detection::intersection::DiscreteIntersection>;
#endif
} // namespace sofa::core::collision
