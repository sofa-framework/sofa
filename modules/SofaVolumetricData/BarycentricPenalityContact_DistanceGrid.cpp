/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaMeshCollision/BarycentricPenalityContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaMeshCollision/BarycentricContactMapper.inl>

#include <SofaVolumetricData/DistanceGridCollisionModel.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::core::collision;

sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleContactClass("default", true);

sofa::core::collision::ContactCreator< BarycentricPenalityContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereContactClass("default", true);
sofa::core::collision::ContactCreator< BarycentricPenalityContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleContactClass("default", true);

} // namespace collision

} // namespace component

} // namespace sofa

