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
#include <sofa/component/collision/response/contact/BarycentricPenalityContact.inl>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>
#include <sofa/component/collision/response/mapper/TetrahedronBarycentricContactMapper.h>

#include <sofa/component/collision/geometry/TetrahedronCollisionModel.h>

namespace sofa::component::collision::response::contact
{

using namespace defaulttype;
using simulation::Node;
using namespace sofa::core::collision;
using namespace sofa::component::collision::geometry;

Creator<Contact::Factory, BarycentricPenalityContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSpherePenalityContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointPenalityContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLinePenalityContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTrianglePenalityContactClass("PenalityContactForceField",true);
Creator<Contact::Factory, BarycentricPenalityContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronPenalityContactClass("PenalityContactForceField",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BarycentricPenalityContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BarycentricPenalityContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BarycentricPenalityContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BarycentricPenalityContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API response::contact::BarycentricPenalityContact<TetrahedronCollisionModel, TetrahedronCollisionModel>;

}  // namespace sofa::component::collision::response::contact
