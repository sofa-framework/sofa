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
#include <sofa/helper/Factory.inl>

#include <sofa/core/collision/Contact.h>
using sofa::core::collision::Contact ;

#include <SofaBaseCollision/CubeModel.h>
#include <SofaBaseCollision/SphereModel.h>

#include <SofaMeshCollision/RigidContactMapper.inl>
#include <SofaMeshCollision/PointModel.h>
#include <SofaMeshCollision/TriangleModel.h>

#include <SofaMeshCollision/BarycentricContactMapper.inl>
#include <SofaMeshCollision/IdentityContactMapper.h>

#include <SofaConstraint/FrictionContact.inl>
#include <SofaConstraint/BarycentricDistanceLMConstraintContact.inl>

#include <SofaMiscCollision/BarycentricStickContact.inl>

#include "components/collision/DistanceGridCollisionModel.h"
#include "components/collision/FFDDistanceGridDiscreteIntersection.h"



namespace sofa
{

namespace component
{

namespace collision
{

int registerDistanceGridCollisionModel()
{
    return 0;
}


///////////////////////////////// BARYCENTRICSTICK /////////////////////////////////////////////////
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleStickContactClass("stick", true);

Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereStickContactClass("stick", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleStickContactClass("stick", true);

/////////////////////////////////////// FRICTION ///////////////////////////////////////////////////
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleFrictionContactClass("FrictionContact", true);

Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereFrictionContactClass("FrictionContact", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleFrictionContactClass("FrictionContact", true);



/////////////////////////////////////// BarycentricDistanceLMConstraint ///////////////////////////////////
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, PointModel> > DistanceGridPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, SphereModel> > DistanceGridSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<RigidDistanceGridCollisionModel, TriangleModel> > DistanceGridTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);

Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, PointModel> > FFDDistanceGridPointDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, SphereModel> > FFDDistanceGridSphereDistanceLMConstraintContactClass("distanceLMConstraint",true);
Creator<Contact::Factory, BarycentricDistanceLMConstraintContact<FFDDistanceGridCollisionModel, TriangleModel> > FFDDistanceGridTriangleDistanceLMConstraintContactClass("distanceLMConstraint",true);


} /// collision
} /// component
} /// sofa
