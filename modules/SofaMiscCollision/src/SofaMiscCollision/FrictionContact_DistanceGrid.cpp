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
#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaVolumetricData/RigidDistanceGridDiscreteIntersection.h>

using namespace sofa::core::collision;

namespace sofa
{

namespace component
{

namespace collision
{


Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridTriangleFrictionContactClass("FrictionContact", true);

Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridTriangleFrictionContactClass("FrictionContact", true);


} // namespace collision

} // namespace component

} // namespace sofa
