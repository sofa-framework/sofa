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

#include <sofa/core/collision/Intersection.h>
#include <sofa/core/collision/IntersectorFactory.h>
#include <SofaBaseCollision/BaseIntTool.h>

namespace sofa::component::collision
{
class SOFA_SOFABASECOLLISION_API DiscreteIntersection : public core::collision::Intersection, public core::collision::BaseIntersector
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

    //Generic case
    template <class Elem1, class Elem2>
    bool testIntersection(Elem1& e1, Elem2& e2)
    {
        return BaseIntTool::testIntersection(e1, e2, this->getAlarmDistance());
    }

    template <class Elem1, class Elem2>
    int computeIntersection(Elem1& e1, Elem2& e2, OutputVector* contacts)
    {
        return BaseIntTool::computeIntersection(e1, e2, e1.getProximity() + e2.getProximity() + getAlarmDistance(), e1.getProximity() + e2.getProximity() + getContactDistance(), contacts);
    }
};


// specializations
// Cube
template <> bool DiscreteIntersection::testIntersection<Cube,Cube>(Cube& sph1, Cube& sph2);
template <> int DiscreteIntersection::computeIntersection<Cube, Cube>(Cube& sph1, Cube& sph2, OutputVector* contacts);

//Sphere
template <> bool DiscreteIntersection::testIntersection<Sphere, Sphere>(Sphere& sph1, Sphere& sph2);
template <> int DiscreteIntersection::computeIntersection<Sphere, Sphere>(Sphere& sph1, Sphere& sph2, OutputVector* contacts);
template <> bool DiscreteIntersection::testIntersection<RigidSphere, RigidSphere>(RigidSphere& sph1, RigidSphere& sph2);
template <> int DiscreteIntersection::computeIntersection<RigidSphere, RigidSphere>(RigidSphere& sph1, RigidSphere& sph2, OutputVector* contacts);
template <> bool DiscreteIntersection::testIntersection<Sphere, RigidSphere>(Sphere& sph1, RigidSphere& sph2);
template <> int DiscreteIntersection::computeIntersection<Sphere, RigidSphere>(Sphere& sph1, RigidSphere& sph2, OutputVector* contacts);
template <> bool DiscreteIntersection::testIntersection<RigidSphere, Sphere>(RigidSphere& sph1, Sphere& sph2);
template <> int DiscreteIntersection::computeIntersection<RigidSphere, Sphere>(RigidSphere& sph1, Sphere& sph2, OutputVector* contacts);

} // namespace sofa::component::collision

namespace sofa::core::collision
{
#if  !defined(SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_CPP)
extern template class SOFA_SOFABASECOLLISION_API IntersectorFactory<component::collision::DiscreteIntersection>;
#endif
} // namespace sofa::core::collision
