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
#define SOFA_COMPONENT_COLLISION_NEWPROXIMITYINTERSECTION_CPP
#include <sofa/helper/system/config.h>
#include <SofaBaseCollision/NewProximityIntersection.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/collision/Intersection.inl>
#include <iostream>
#include <algorithm>

namespace sofa
{

namespace core
{
    namespace collision
    {
        template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::NewProximityIntersection>;
    }
}

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

int NewProximityIntersectionClass = core::RegisterObject("Optimized Proximity Intersection based on Triangle-Triangle tests, ignoring Edge-Edge cases")
        .add< NewProximityIntersection >()
        ;

NewProximityIntersection::NewProximityIntersection()
    : BaseProximityIntersection()
    , useLineLine(initData(&useLineLine, false, "useLineLine", "Line-line collision detection enabled"))
{
}

void NewProximityIntersection::init()
{
    intersectors.add<CubeCollisionModel, CubeCollisionModel, NewProximityIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, NewProximityIntersection>(this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, NewProximityIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,SphereCollisionModel<sofa::defaulttype::Vec3Types>, NewProximityIntersection> (this);
    intersectors.add<OBBCollisionModel<sofa::defaulttype::Rigid3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>, NewProximityIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>, NewProximityIntersection> (this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,RigidSphereModel, NewProximityIntersection> (this);
    intersectors.add<RigidSphereModel,OBBCollisionModel<sofa::defaulttype::Rigid3Types>, NewProximityIntersection> (this);

    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, NewProximityIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,SphereCollisionModel<sofa::defaulttype::Vec3Types>, NewProximityIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>, NewProximityIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,RigidSphereModel, NewProximityIntersection> (this);

    IntersectorFactory::getInstance()->addIntersectors(this);

	BaseProximityIntersection::init();
}

} // namespace collision

} // namespace component

} // namespace sofa

