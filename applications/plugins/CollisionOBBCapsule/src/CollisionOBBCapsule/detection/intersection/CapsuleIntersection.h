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
#include <CollisionOBBCapsule/config.h>

#include <sofa/core/collision/Intersection.h>

#include <sofa/component/collision/detection/intersection/DiscreteIntersection.h>
#include <sofa/component/collision/detection/intersection/MinProximityIntersection.h>
#include <sofa/component/collision/detection/intersection/NewProximityIntersection.h>
#include <sofa/component/collision/detection/intersection/MeshNewProximityIntersection.h>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>
#include <CollisionOBBCapsule/detection/intersection/BaseIntTool.h>
#include <CollisionOBBCapsule/detection/intersection/MeshIntTool.h>

namespace collisionobbcapsule::detection::intersection
{
using sofa::component::collision::detection::intersection::DiscreteIntersection;
using sofa::component::collision::detection::intersection::NewProximityIntersection;

class COLLISIONOBBCAPSULE_API CapsuleDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    CapsuleDiscreteIntersection(DiscreteIntersection* intersection);

    template <class Elem1, class Elem2>
    int computeIntersection(Elem1& e1, Elem2& e2, OutputVector* contacts, const core::collision::Intersection* intersection) {
        return BaseIntTool::computeIntersection(e1,
            e2, 
            e1.getProximity() + e2.getProximity() + intersection->getAlarmDistance(),
            e1.getProximity() + e2.getProximity() + intersection->getContactDistance(),
            contacts);
    }

    template <class Elem1, class Elem2>
    bool testIntersection(Elem1& e1, Elem2& e2, const core::collision::Intersection* intersection) {
        return BaseIntTool::testIntersection(e1, e2, intersection->getAlarmDistance());
    }

};


class COLLISIONOBBCAPSULE_API CapsuleMeshDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    CapsuleMeshDiscreteIntersection(NewProximityIntersection* intersection);

    template <class Elem1, class Elem2>
    int computeIntersection(Elem1& e1, Elem2& e2, OutputVector* contacts, const core::collision::Intersection* intersection) {
        return MeshIntTool::computeIntersection(e1,
            e2,
            e1.getProximity() + e2.getProximity() + intersection->getAlarmDistance(),
            e1.getProximity() + e2.getProximity() + intersection->getContactDistance(),
            contacts);
    }

    template <class Elem1, class Elem2>
    bool testIntersection(Elem1& e1, Elem2& e2, const core::collision::Intersection* intersection) {
        return BaseIntTool::testIntersection(e1, e2, intersection->getAlarmDistance());
    }

    bool testIntersection(Capsule&, Triangle&, const core::collision::Intersection*) { return true; }
    bool testIntersection(Capsule&, Line&, const core::collision::Intersection*) { return true; }

};

} // namespace collisionobbcapsule::detection::intersection
