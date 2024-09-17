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
#include <sofa/component/constraint/lagrangian/model/BaseContactLagrangianConstraint.inl>
#include <sofa/component/collision/response/contact/AugmentedLagrangianResponse.inl>
#include <sofa/component/collision/response/contact/BaseUnilateralContactResponse.inl>

#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>

namespace sofa::component::collision::response::contact
{

using namespace defaulttype;
using namespace sofa::helper;
using namespace sofa::component::collision::geometry;
using simulation::Node;

Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<RigidSphereModel, RigidSphereModel> > RigidSphereRigidSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > SphereRigidSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > LineRigidSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel> > TriangleRigidSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<sofa::core::collision::Contact::Factory, AugmentedLagrangianResponse<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > RigidSpherePointAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<RigidSphereModel, RigidSphereModel,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BaseUnilateralContactResponse<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<RigidSphereModel, RigidSphereModel>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<SphereCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<LineCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, RigidSphereModel>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API AugmentedLagrangianResponse<RigidSphereModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>;

} //namespace sofa::component::collision::response::contact
