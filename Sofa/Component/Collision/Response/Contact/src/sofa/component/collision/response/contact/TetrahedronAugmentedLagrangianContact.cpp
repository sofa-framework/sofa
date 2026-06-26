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
#include <sofa/component/collision/response/contact/config.h>
#include <sofa/component/collision/response/contact/AugmentedLagrangianResponse.inl>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/TetrahedronBarycentricContactMapper.h>
#include <sofa/component/collision/geometry/TetrahedronCollisionModel.h>
#include <sofa/component/collision/response/contact/BaseUnilateralContactResponse.inl>

using namespace sofa::core::collision;

namespace sofa::component::collision::response::contact
{

using namespace sofa::component::collision::geometry;

Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSphereAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLineAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTriangleAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);

Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSpherePenalityAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointPenalityAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLinePenalityAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTrianglePenalityAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);
Creator<Contact::Factory, AugmentedLagrangianResponse<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronPenalityAugmentedLagrangianResponseClass("AugmentedLagrangianResponseConstraint",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, TetrahedronCollisionModel,constraint::lagrangian::model::AugmentedLagrangianContactParameters>;

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::AugmentedLagrangianResponse<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::AugmentedLagrangianResponse<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::AugmentedLagrangianResponse<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::AugmentedLagrangianResponse<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::AugmentedLagrangianResponse<TetrahedronCollisionModel, TetrahedronCollisionModel>;

}  // namespace sofa::component::collision::response::contact
