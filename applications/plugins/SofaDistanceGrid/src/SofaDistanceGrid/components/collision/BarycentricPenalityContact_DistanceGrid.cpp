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

#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>

#include "DistanceGridCollisionModel.h"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::core::collision;
using namespace sofa::component::collision::response::contact;
using namespace sofa::component::collision::geometry;

Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridPointContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridSphereContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<RigidDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridTriangleContactClass("PenalityContactForceField", true);

Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridPointContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridSphereContactClass("PenalityContactForceField", true);
Creator<Contact::Factory, BarycentricPenalityContact<FFDDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridTriangleContactClass("PenalityContactForceField", true);

} // namespace collision

} // namespace component

} // namespace sofa

