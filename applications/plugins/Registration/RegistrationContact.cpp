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

Creator<Contact::Factory, RegistrationContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > SphereSphereRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<SphereCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > SpherePointRegistrationContactClass("RegistrationContactForceField",true);
//Creator<Contact::Factory, RegistrationContact<SphereTreeModel, SphereTreeModel> > SphereTreeSphereTreeRegistrationContactClass("RegistrationContactForceField", true);
//Creator<Contact::Factory, RegistrationContact<SphereTreeModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > SphereTreeTriangleContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<PointCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > PointPointRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > LinePointRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > LineLineRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<LineCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > LineSphereRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleSphereRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TrianglePointRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleLineRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TriangleCollisionModel<sofa::defaulttype::Vec3Types>, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TriangleTriangleRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronSphereRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronPointRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, LineCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronLineRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > TetrahedronTriangleRegistrationContactClass("RegistrationContactForceField",true);
Creator<Contact::Factory, RegistrationContact<TetrahedronCollisionModel, TetrahedronCollisionModel> > TetrahedronTetrahedronRegistrationContactClass("RegistrationContactForceField",true);

#if REGISTRATION_HAVE_SOFADISTANCEGRID
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridRegistrationContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridPointRegistrationContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridSphereRegistrationContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<RigidDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridTriangleRegistrationContactClass("RegistrationContactForceField", true);

Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridRegistrationContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridRegistrationContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridPoinRegistrationtContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridSphereRegistrationContactClass("RegistrationContactForceField", true);
Creator<Contact::Factory, RegistrationContact<FFDDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridTriangleRegistrationContactClass("RegistrationContactForceField", true);
#endif

} // namespace collision

} // namespace component

} // namespace sofa

