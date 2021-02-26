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
#define SOFA_COMPONENT_COLLISION_BARYCENTRICPENALITYCONTACT_CPP
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaBaseCollision/RigidCapsuleModel.h>

namespace sofa::component::collision
{

using namespace core::collision;
using simulation::Node;

Creator<Contact::Factory, BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLinePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLinePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTrianglePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleTrianglePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleLinePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleCapsulePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > CapsuleRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > OBBOBBPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleOBBPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > SphereOBBPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidSphereOBBPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > TriangleOBBPenalityContactClass("penality",true);

Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleTrianglePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleLinePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleRigidCapsulePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleRigidCapsulePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > RigidCapsuleSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > RigidCapsuleRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > RigidCapsuleOBBPenalityContactClass("penality",true);

Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > CylinderCylinderPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > CylinderTrianglePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>> > CylinderRigidCapsulePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>> > CapsuleCylinderPenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > CylinderSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel> > CylinderRigidSpherePenalityContactClass("penality",true);
Creator<Contact::Factory, BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>> > CylinderOBBPenalityContactClass("penality",true);



template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<RigidSphereModel, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<OBBCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<RigidSphereModel, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, CylinderCollisionModel<sofa::defaulttype::Rigid3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, RigidSphereModel>;
template class SOFA_SOFAMESHCOLLISION_API BarycentricPenalityContact<CylinderCollisionModel<sofa::defaulttype::Rigid3Types>, OBBCollisionModel<sofa::defaulttype::Rigid3Types>>;

} //namespace sofa::component::collision
