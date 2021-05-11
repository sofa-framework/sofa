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
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_CPP

#include <SofaBaseCollision/DiscreteIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>

#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/BaseIntTool.h>

namespace sofa::core::collision
{
    template class SOFA_SOFABASECOLLISION_API IntersectorFactory<component::collision::DiscreteIntersection>;
} // namespace sofa::core::collision

namespace sofa::component::collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

int DiscreteIntersectionClass = core::RegisterObject("TODO-DiscreteIntersectionClass")
        .add< DiscreteIntersection >()
        ;

DiscreteIntersection::DiscreteIntersection()
{
    intersectors.add<CubeCollisionModel,       CubeCollisionModel,         DiscreteIntersection> (this);

    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>,     SphereCollisionModel<sofa::defaulttype::Vec3Types>,       DiscreteIntersection> (this);

    intersectors.add<RigidSphereModel,RigidSphereModel,DiscreteIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>,RigidSphereModel, DiscreteIntersection> (this);

	IntersectorFactory::getInstance()->addIntersectors(this);
}

/// Return the intersector class handling the given pair of collision models, or nullptr if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels)
{
    return intersectors.get(object1, object2, swapModels);
}

template<>
bool DiscreteIntersection::testIntersection(Cube& cube1, Cube& cube2)
{
    const SReal alarmDist = this->getAlarmDistance();

    if (cube1 == cube2)
    {
        if (cube1.getConeAngle() < M_PI / 2)
            return false;
        else
            return true;
    }

    const defaulttype::Vector3& minVect1 = cube1.minVect();
    const defaulttype::Vector3& minVect2 = cube2.minVect();
    const defaulttype::Vector3& maxVect1 = cube1.maxVect();
    const defaulttype::Vector3& maxVect2 = cube2.maxVect();

    for (int i = 0; i < 3; i++)
    {
        if (minVect1[i] > maxVect2[i] + alarmDist || minVect2[i] > maxVect1[i] + alarmDist)
            return false;
    }

    return true;
}

template<>
int DiscreteIntersection::computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts)
{
    SOFA_UNUSED(cube1);
    SOFA_UNUSED(cube2);
    SOFA_UNUSED(contacts);

    return 0;
}

namespace // anonymous
{
    template<class SphereType1, class SphereType2>
    bool testIntersectionSphere(SphereType1& sph1, SphereType2& sph2, const SReal alarmDist)
    {
        const auto r = sph1.r() + sph2.r() + alarmDist;
        return (sph1.center() - sph2.center()).norm2() <= r * r;
    }

    template<class SphereType1, class SphereType2>
    int computeIntersectionSphere(SphereType1& sph1, SphereType2& sph2, DiscreteIntersection::OutputVector* contacts, const SReal alarmDist, const SReal contactDist)
    {    
        SReal r = sph1.r() + sph2.r();
        SReal myAlarmDist = alarmDist + r;
        defaulttype::Vector3 dist = sph2.center() - sph1.center();
        SReal norm2 = dist.norm2();

        if (norm2 > myAlarmDist * myAlarmDist)
            return 0;

        contacts->resize(contacts->size() + 1);
        DetectionOutput* detection = &*(contacts->end() - 1);
        SReal distSph1Sph2 = helper::rsqrt(norm2);
        detection->normal = dist / distSph1Sph2;
        detection->point[0] = sph1.getContactPointByNormal(-detection->normal);
        detection->point[1] = sph2.getContactPointByNormal(detection->normal);

        detection->value = distSph1Sph2 - r - contactDist;
        detection->elem.first = sph1;
        detection->elem.second = sph2;
        detection->id = (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex();

        return 1;
    }
} // anonymous

template <> 
bool DiscreteIntersection::testIntersection<Sphere, Sphere>(Sphere& sph1, Sphere& sph2)
{
    return testIntersectionSphere(sph1, sph2, this->getAlarmDistance());
}

template <> 
int DiscreteIntersection::computeIntersection<Sphere, Sphere>(Sphere& sph1, Sphere& sph2, OutputVector* contacts)
{
    return computeIntersectionSphere(sph1, sph2, contacts, this->getAlarmDistance(), this->getContactDistance());
}

template <> 
bool DiscreteIntersection::testIntersection<RigidSphere, RigidSphere>(RigidSphere& sph1, RigidSphere& sph2)
{
    return testIntersectionSphere(sph1, sph2, this->getAlarmDistance());
}

template <> 
int DiscreteIntersection::computeIntersection<RigidSphere, RigidSphere>(RigidSphere& sph1, RigidSphere& sph2, OutputVector* contacts)
{
    return computeIntersectionSphere(sph1, sph2, contacts, this->getAlarmDistance(), this->getContactDistance());
}

template <> 
bool DiscreteIntersection::testIntersection<Sphere, RigidSphere>(Sphere& sph1, RigidSphere& sph2)
{
    return testIntersectionSphere(sph1, sph2, this->getAlarmDistance());
}

template <> 
int DiscreteIntersection::computeIntersection<Sphere, RigidSphere>(Sphere& sph1, RigidSphere& sph2, OutputVector* contacts)
{
    return computeIntersectionSphere(sph1, sph2, contacts, this->getAlarmDistance(), this->getContactDistance());
}

template <> 
bool DiscreteIntersection::testIntersection<RigidSphere, Sphere>(RigidSphere& sph1, Sphere& sph2)
{
    return testIntersectionSphere(sph1, sph2, this->getAlarmDistance());
}

template <> 
int DiscreteIntersection::computeIntersection<RigidSphere, Sphere>(RigidSphere& sph1, Sphere& sph2, OutputVector* contacts)
{
    return computeIntersectionSphere(sph1, sph2, contacts, this->getAlarmDistance(), this->getContactDistance());
}


} // namespace sofa::component::collision
