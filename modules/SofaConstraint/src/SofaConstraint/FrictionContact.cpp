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
#include <SofaConstraint/FrictionContact.inl>

#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>

namespace sofa::component::collision
{

using namespace defaulttype;
using namespace sofa::helper;
using simulation::Node;

Creator<sofa::core::collision::Contact::Factory, FrictionContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsuleFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTriangleFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsuleFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleRigidCapsuleFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTriangleFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSphereFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBFrictionContactClass("LagrangeMultipliers",true);
Creator<sofa::core::collision::Contact::Factory, FrictionContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSphereFrictionContactClass("LagrangeMultipliers",true);


} //namespace sofa::component::collision
