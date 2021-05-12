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
#include <SofaBaseCollision/config.h>

#include <SofaBaseCollision/BaseProximityIntersection.h>

namespace sofa::component::collision
{

class SOFA_SOFABASECOLLISION_API NewProximityIntersection : public BaseProximityIntersection
{
public:
    SOFA_CLASS(NewProximityIntersection,BaseProximityIntersection);

    Data<bool> useLineLine; ///< Line-line collision detection enabled

    typedef core::collision::IntersectorFactory<NewProximityIntersection> IntersectorFactory;

    void init() override;

    static inline int doIntersectionPointPoint(SReal dist2, const defaulttype::Vector3& p, const defaulttype::Vector3& q, OutputVector* contacts, int id);


    bool testIntersection(Cube& cube1, Cube& cube2);
    int computeIntersection(Cube& cube1, Cube& cube2, OutputVector* contacts);

    bool testIntersection(Sphere& sph1, Sphere& sph2);
    int computeIntersection(Sphere& sph1, Sphere& sph2, OutputVector* contacts);
    bool testIntersection(Sphere& sph1, RigidSphere& sph2);
    int computeIntersection(Sphere& sph1, RigidSphere& sph2, OutputVector* contacts);
    bool testIntersection(RigidSphere& sph1, RigidSphere& sph2);
    int computeIntersection(RigidSphere& sph1, RigidSphere& sph2, OutputVector* contacts);

protected:
    NewProximityIntersection();

    template<class SphereType1, class SphereType2>
    bool testIntersectionSphere(SphereType1& sph1, SphereType2& sph2, const SReal alarmDist)
    {
        OutputVector contacts;
        const double alarmDist2 = alarmDist + sph1.r() + sph2.r();
        int n = doIntersectionPointPoint(alarmDist2 * alarmDist2, sph1.center(), sph2.center(), &contacts, -1);
        return n > 0;
    }

    template<class SphereType1, class SphereType2>
    int computeIntersectionSphere(SphereType1& sph1, SphereType2& sph2, DiscreteIntersection::OutputVector* contacts, const SReal alarmDist, const SReal contactDist)
    {
        const double alarmDist2 = alarmDist + sph1.r() + sph2.r();
        int n = doIntersectionPointPoint(alarmDist2 * alarmDist2, sph1.center(), sph2.center(), contacts, (sph1.getCollisionModel()->getSize() > sph2.getCollisionModel()->getSize()) ? sph1.getIndex() : sph2.getIndex());
        if (n > 0)
        {
            const double contactDist2 = contactDist + sph1.r() + sph2.r();
            for (OutputVector::iterator detection = contacts->end() - n; detection != contacts->end(); ++detection)
            {
                detection->elem = std::pair<core::CollisionElementIterator, core::CollisionElementIterator>(sph1, sph2);
                detection->value -= contactDist2;
            }
        }
        return n;
    }

};

} // namespace sofa::component::collision

namespace sofa::core::collision
{
#if  !defined(SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_CPP)
extern template class SOFA_SOFABASECOLLISION_API IntersectorFactory<component::collision::NewProximityIntersection>;
#endif

} // namespace sofa::core::collision
