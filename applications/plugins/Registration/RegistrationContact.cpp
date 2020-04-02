/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "RegistrationContact.inl"
#include "RegistrationContactForceField.inl"

#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaMeshCollision/IdentityContactMapper.inl>
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMiscCollision/TetrahedronModel.h>

#if REGISTRATION_HAVE_SOFADISTANCEGRID
#include <SofaDistanceGrid/components/collision/DistanceGridCollisionModel.h>
#endif


namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;
using simulation::Node;

Creator<Contact::Factory, RegistrationContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointRegistrationContactClass("registration",true);
//Creator<Contact::Factory, RegistrationContact<SphereTreeModel, SphereTreeModel> > SphereTreeSphereTreeRegistrationContactClass("registration", true);
//Creator<Contact::Factory, RegistrationContact<SphereTreeModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > SphereTreeTriangleContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSphereRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLineRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTriangleRegistrationContactClass("registration",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronRegistrationContactClass("registration",true);

#if REGISTRATION_HAVE_SOFADISTANCEGRID
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridRegistrationContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridPointRegistrationContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridSphereRegistrationContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridTriangleRegistrationContactClass("registration", true);

Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridRegistrationContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridRegistrationContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridPoinRegistrationtContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridSphereRegistrationContactClass("registration", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridTriangleRegistrationContactClass("registration", true);
#endif

} // namespace collision

} // namespace component

} // namespace sofa

