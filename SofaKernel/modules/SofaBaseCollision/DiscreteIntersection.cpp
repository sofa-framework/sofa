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
#include <sofa/helper/system/config.h>

#include <SofaBaseCollision/DiscreteIntersection.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/collision/Intersection.inl>

#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/CubeModel.h>
#include <SofaBaseCollision/CapsuleModel.h>
#include <SofaBaseCollision/OBBModel.h>
#include <SofaBaseCollision/BaseIntTool.h>
#include <SofaBaseCollision/RigidCapsuleModel.h>

namespace sofa
{

namespace core
{
    namespace collision
    {
        template class SOFA_BASE_COLLISION_API IntersectorFactory<component::collision::DiscreteIntersection>;
    }
}


namespace component
{

namespace collision
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

    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, DiscreteIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,SphereCollisionModel<sofa::defaulttype::Vec3Types>, DiscreteIntersection> (this);

    intersectors.add<OBBCollisionModel<sofa::defaulttype::Rigid3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>,DiscreteIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>, DiscreteIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>,DiscreteIntersection>(this);

    intersectors.add<RigidSphereModel,RigidSphereModel,DiscreteIntersection>(this);
    intersectors.add<SphereCollisionModel<sofa::defaulttype::Vec3Types>,RigidSphereModel, DiscreteIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,RigidSphereModel,DiscreteIntersection>(this);
    intersectors.add<RigidSphereModel,OBBCollisionModel<sofa::defaulttype::Rigid3Types>,DiscreteIntersection>(this);

    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>,CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, DiscreteIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, DiscreteIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,SphereCollisionModel<sofa::defaulttype::Vec3Types>, DiscreteIntersection> (this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,OBBCollisionModel<sofa::defaulttype::Rigid3Types>,DiscreteIntersection>(this);
    intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>,RigidSphereModel,DiscreteIntersection>(this);

	IntersectorFactory::getInstance()->addIntersectors(this);
}

/// Return the intersector class handling the given pair of collision models, or nullptr if not supported.
ElementIntersector* DiscreteIntersection::findIntersector(core::CollisionModel* object1, core::CollisionModel* object2, bool& swapModels)
{
    return intersectors.get(object1, object2, swapModels);
}


} // namespace collision

} // namespace component

} // namespace sofa

