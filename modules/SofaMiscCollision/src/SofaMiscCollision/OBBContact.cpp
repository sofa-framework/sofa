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
#include <SofaMiscCollision/OBBModel.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseCollision/CylinderModel.h>
#include <SofaMeshCollision/TriangleModel.h>
#include <SofaUserInteraction/RayModel.h>
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaMiscCollision/OBBContactMapper.h>
#include <SofaConstraint/FrictionContact.inl>
#include <SofaUserInteraction/RayContact.h>

using namespace sofa::core::collision;

namespace sofa::component::collision
{

Creator<Contact::Factory, BarycentricPenalityContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CylinderOBBPenalityContactClass("default", true);

Creator<Contact::Factory, FrictionContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBFrictionContactClass("FrictionContact", true);

Creator<Contact::Factory, RayContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RayRigidBoxContactClass("ray", true); //cast not wroking

template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class SOFA_MISC_COLLISION_API FrictionContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_MISC_COLLISION_API FrictionContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class SOFA_MISC_COLLISION_API RayContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

} // namespace sofa::component::collision
