/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <SofaConstraint/BarycentricDistanceLMConstraintContact.inl>
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaVolumetricData/RigidDistanceGridDiscreteIntersection.h>


using namespace sofa::defaulttype;
using namespace sofa::core::collision;

namespace sofa
{

namespace component
{

namespace collision
{

Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);


Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);


} // namespace collision

} // namespace component

} // namespace sofa

