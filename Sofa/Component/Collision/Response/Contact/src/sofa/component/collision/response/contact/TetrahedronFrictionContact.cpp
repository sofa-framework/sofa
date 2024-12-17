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
#include <sofa/component/collision/response/contact/FrictionContact.inl>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/TetrahedronBarycentricContactMapper.h>
#include <sofa/component/collision/geometry/TetrahedronModel.h>
#include <sofa/component/collision/response/contact/BaseUnilateralContactResponse.inl>

using namespace sofa::core::collision;

namespace sofa::component::collision::response::contact
{

using namespace sofa::component::collision::geometry;

Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSphereFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLineFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTriangleFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronFrictionContactClass("FrictionContactConstraint",true);

Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSpherePenalityFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointPenalityFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLinePenalityFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTrianglePenalityFrictionContactClass("FrictionContactConstraint",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronPenalityFrictionContactClass("FrictionContactConstraint",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::UnilateralLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::UnilateralLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::UnilateralLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>,constraint::lagrangian::model::UnilateralLagrangianContactParameters>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BaseUnilateralContactResponse<TetrahedronCollisionModel, TetrahedronCollisionModel,constraint::lagrangian::model::UnilateralLagrangianContactParameters>;

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::FrictionContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::FrictionContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::FrictionContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::FrictionContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::FrictionContact<TetrahedronCollisionModel, TetrahedronCollisionModel>;

}  // namespace sofa::component::collision::response::contact
