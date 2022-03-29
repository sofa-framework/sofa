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
#include <CollisionOBBCapsule/model/OBBModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/CylinderModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <CollisionOBBCapsule/response/mapper/OBBContactMapper.h>
#include <SofaConstraint/FrictionContact.inl>
#include <SofaUserInteraction/RayContact.h>


namespace collisionobbcapsule::response::contact
{

using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;
using namespace sofa::component::collision::response::contact;
using namespace collisionobbcapsule::model;

Creator<Contact::Factory, BarycentricPenalityContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBPenalityContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CylinderOBBPenalityContactClass("PenalityContactForceField", true);

Creator<Contact::Factory, FrictionContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBFrictionContactClass("FrictionContactConstraint", true);

Creator<Contact::Factory, RayContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RayRigidBoxContactClass("RayContact", true); //cast not working

template class COLLISIONOBB_API response::contact::BarycentricPenalityContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::BarycentricPenalityContact<collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::BarycentricPenalityContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::BarycentricPenalityContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::BarycentricPenalityContact<collision::geometry::CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class COLLISIONOBB_API response::contact::FrictionContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::FrictionContact<collision::geometry::SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::FrictionContact<collision::geometry::TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class COLLISIONOBB_API response::contact::FrictionContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class COLLISIONOBBCAPSULE_API response::contact::RayContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

} // namespace collisionobbcapsule::response::contact
