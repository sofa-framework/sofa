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
#include <SofaMiscCollision/config.h>

#include <sofa/core/collision/Intersection.h>

#include <SofaBaseCollision/DiscreteIntersection.h>
#include <SofaMiscCollision/OBBModel.h>
#include <SofaMiscCollision/BaseIntTool.h>
#include <SofaMiscCollision/MeshIntTool.h>
#include <SofaMeshCollision/MeshNewProximityIntersection.h>
#include <SofaUserInteraction/RayModel.h>

namespace sofa::component::collision
{

class SOFA_MISC_COLLISION_API RigidDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    RigidDiscreteIntersection(DiscreteIntersection* object);

    template <class Elem1, class Elem2>
    int computeIntersection(Elem1& e1, Elem2& e2, OutputVector* contacts) {
        return BaseIntTool::computeIntersection(e1,
            e2,
            e1.getProximity() + e2.getProximity() + intersection->getAlarmDistance(),
            e1.getProximity() + e2.getProximity() + intersection->getContactDistance(),
            contacts);
    }

    template <class Elem1, class Elem2>
    bool testIntersection(Elem1& e1, Elem2& e2) {
        return BaseIntTool::testIntersection(e1, e2, intersection->getAlarmDistance());
    }

    bool testIntersection(Ray& /*rRay*/, OBB& /*rOBB*/);
    int computeIntersection(Ray& rRay, OBB& rObb, OutputVector* contacts);

protected:
    DiscreteIntersection* intersection;

};


class SOFA_MISC_COLLISION_API RigidMeshDiscreteIntersection : public core::collision::BaseIntersector
{
    typedef DiscreteIntersection::OutputVector OutputVector;

public:
    RigidMeshDiscreteIntersection(NewProximityIntersection* object);

    template <class Elem1, class Elem2>
    int computeIntersection(Elem1& e1, Elem2& e2, OutputVector* contacts) {
        return MeshIntTool::computeIntersection(e1,
            e2,
            e1.getProximity() + e2.getProximity() + intersection->getAlarmDistance(),
            e1.getProximity() + e2.getProximity() + intersection->getContactDistance(),
            contacts);
    }

    template <class Elem1, class Elem2>
    bool testIntersection(Elem1& e1, Elem2& e2) {
        return BaseIntTool::testIntersection(e1, e2, intersection->getAlarmDistance());
    }

protected:
    NewProximityIntersection* intersection;

};


} // namespace sofa::component::collision
