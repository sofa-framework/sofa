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
#include <SofaMiscCollision/CapsuleModel.h>
#include <SofaMiscCollision/RigidCapsuleModel.h>
#include <SofaMiscCollision/OBBModel.h>
#include <SofaMiscCollision/OBBContactMapper.h>
#include <SofaMiscCollision/CapsuleContactMapper.h>
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaConstraint/FrictionContact.inl>

using namespace sofa::core::collision;

namespace sofa::component::collision
{

Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleLinePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsulePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSpherePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTriangleContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleLinePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsulePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleRigidCapsulePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSpherePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSpherePenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleCylinderPenalityContactClass("default", true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleCylinderPenalityContactClass("default", true);

Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsuleFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsuleFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleRigidCapsuleFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTriangleFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBFrictionContactClass("FrictionContact", true);

} // namespace sofa::component::collision
