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
#include <SofaMiscCollision/CapsuleIntersection.h>

#include <sofa/core/collision/IntersectorFactory.h>
#include <sofa/core/collision/Intersection.inl>

#include <SofaMiscCollision/CapsuleModel.h>
#include <SofaMiscCollision/RigidCapsuleModel.h>
#include <SofaMiscCollision/OBBModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaUserInteraction/FixParticlePerformer.h>

namespace sofa::component::collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

IntersectorCreator<DiscreteIntersection, CapsuleDiscreteIntersection> CapsuleDiscreteIntersectors("Capsule");
IntersectorCreator<NewProximityIntersection, CapsuleMeshDiscreteIntersection> CapsuleMeshDiscreteIntersectors("CapsuleMesh");

CapsuleDiscreteIntersection::CapsuleDiscreteIntersection(DiscreteIntersection* object)
    : intersection(object)
{
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleDiscreteIntersection>(this);
    
    intersection->intersectors.ignore<RayCollisionModel, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>>();
    intersection->intersectors.ignore<RayCollisionModel, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>();
}

CapsuleMeshDiscreteIntersection::CapsuleMeshDiscreteIntersection(NewProximityIntersection* object)
    : intersection(object)
{

    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleMeshDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleMeshDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleMeshDiscreteIntersection>(this);
    intersection->intersectors.add<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleMeshDiscreteIntersection>(this);
}

// add CapsuleModel to the list of supported collision models for FixParticlePerformer
using FixParticlePerformer3d = sofa::component::collision::FixParticlePerformer<defaulttype::Vec3Types>;

int capsuleFixParticle = FixParticlePerformer3d::RegisterSupportedModel<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>>(
    []
(sofa::core::sptr<sofa::core::CollisionModel> model, const Index idx, type::vector<Index>& points, FixParticlePerformer3d::Coord& fixPoint)
    {
        auto* caps = dynamic_cast<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>*>(model.get());

        if (!caps)
            return false;

        fixPoint = (caps->point1(idx) + caps->point2(idx))/2.0;
        points.push_back(caps->point1Index(idx));
        points.push_back(caps->point2Index(idx));

        return true;
    }
);


} // namespace sofa::component::collision
