/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleFrictionContactClass("FrictionContact", true);

Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleFrictionContactClass("FrictionContact", true);


} // namespace collision

} // namespace component

} // namespace sofa
