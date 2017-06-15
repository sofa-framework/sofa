/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_COLLISION_DISCRETEINTERSECTION_CPP
#include <sofa/helper/system/config.h>
#include <sofa/helper/FnDispatcher.inl>
#include <SofaBaseCollision/DiscreteIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>
#include <sofa/helper/proximity.h>
#include <iostream>
#include <algorithm>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

SOFA_DECL_CLASS(DiscreteIntersection)

int DiscreteIntersectionClass = core::RegisterObject("TODO-DiscreteIntersectionClass")
        .add< DiscreteIntersection >()
        ;


DiscreteIntersection::DiscreteIntersection()
{
    intersectors.add<CubeModel,       CubeModel,         DiscreteIntersection> (this);

    intersectors.add<SphereModel,     SphereModel,       DiscreteIntersection> (this);

    intersectors.add<CapsuleModel,CapsuleModel, DiscreteIntersection> (this);
    intersectors.add<CapsuleModel,SphereModel, DiscreteIntersection> (this);

    intersectors.add<OBBModel,OBBModel,DiscreteIntersection>(this);
    intersectors.add<SphereModel,OBBModel, DiscreteIntersection> (this);
    intersectors.add<CapsuleModel,OBBModel,DiscreteIntersection>(this);

    intersectors.add<RigidSphereModel,RigidSphereModel,DiscreteIntersection>(this);
    intersectors.add<SphereModel,RigidSphereModel, DiscreteIntersection> (this);
    intersectors.add<CapsuleModel,RigidSphereModel,DiscreteIntersection>(this);
    intersectors.add<RigidSphereModel,OBBModel,DiscreteIntersection>(this);

    intersectors.add<CapsuleModel,RigidCapsuleModel, DiscreteIntersection> (this);
    intersectors.add<RigidCapsuleModel,RigidCapsuleModel, DiscreteIntersection> (this);
    intersectors.add<RigidCapsuleModel,SphereModel, DiscreteIntersection> (this);
    intersectors.add<RigidCapsuleModel,OBBModel,DiscreteIntersection>(this);
    intersectors.add<RigidCapsuleModel,RigidSphereModel,DiscreteIntersection>(this);

    IntersectorFactory::getInstance()->addIntersectors(this);
}

/// Return the intersector class handling the given pair of collision models, or NULL if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels)
{
    return intersectors.get(object1, object2, swapModels);
}


} // namespace collision

} // namespace component

namespace core
{
namespace collision
{
template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::DiscreteIntersection>;
}
}

} // namespace sofa

