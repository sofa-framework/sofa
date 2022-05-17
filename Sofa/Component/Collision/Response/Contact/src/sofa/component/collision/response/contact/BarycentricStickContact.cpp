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
#include <sofa/component/collision/response/contact/BarycentricStickContact.inl>

#include <sofa/component/collision/response/mapper/BarycentricContactMapper.h>
#include <sofa/component/collision/response/mapper/IdentityContactMapper.h>

using namespace sofa::core::collision ;

namespace sofa::component::collision::response::contact
{

using namespace sofa::component::collision::geometry;

Creator<Contact::Factory, BarycentricStickContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineStickContactClass("StickContactForceField",true);
Creator<Contact::Factory, BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleStickContactClass("StickContactForceField",true);

template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>>;
template class SOFA_COMPONENT_COLLISION_RESPONSE_CONTACT_API BarycentricStickContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>>;

}  // namespace sofa::component::collision::response::contact
