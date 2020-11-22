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
#include <SofaMiscCollision/TetrahedronModel.h>

using namespace sofa::core::collision;

namespace sofa
{

namespace component
{

namespace collision
{

Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSphereFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLineFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTriangleFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronFrictionContactClass("FrictionContact",true);

Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSpherePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointPenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLinePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTrianglePenalityFrictionContactClass("FrictionContact",true);
Creator<Contact::Factory, FrictionContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronPenalityFrictionContactClass("FrictionContact",true);

} // namespace collision

} // namespace component

} // namespace sofa
