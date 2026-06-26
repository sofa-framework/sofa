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
#include <sofa/helper/Factory.inl>

#include <sofa/core/collision/Contact.h>
using sofa::core::collision::Contact ;

#include <sofa/component/collision/geometry/CubeCollisionModel.h>
#include <sofa/component/collision/geometry/SphereCollisionModel.h>
#include <sofa/component/collision/geometry/PointCollisionModel.h>
#include <sofa/component/collision/geometry/TriangleCollisionModel.h>

#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>
#include <sofa/component/collision/response/mapper/IdentityContactMapper.inl>

#include <sofa/component/collision/response/contact/FrictionContact.inl>
#include <sofa/component/collision/response/contact/BarycentricStickContact.inl>
#include <sofa/component/collision/response/contact/BaseUnilateralContactResponse.inl>

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

using namespace sofa::component::collision::response::contact;
using namespace sofa::component::collision::geometry;

///////////////////////////////// BARYCENTRICSTICK /////////////////////////////////////////////////
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridPointStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridSphereStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<RigidDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridTriangleStickContactClass("StickContactForceField", true);

Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridPointStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridSphereStickContactClass("StickContactForceField", true);
Creator<Contact::Factory, BarycentricStickContact<FFDDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridTriangleStickContactClass("StickContactForceField", true);

/////////////////////////////////////// FRICTION ///////////////////////////////////////////////////
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > DistanceGridDistanceGridFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridPointFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridSphereFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<RigidDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > DistanceGridTriangleFrictionContactClass("FrictionContactConstraint", true);

Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, FFDDistanceGridCollisionModel> > FFDDistanceGridFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, RigidDistanceGridCollisionModel> > FFDDistanceGridRigidDistanceGridFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, PointCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridPointFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, SphereCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridSphereFrictionContactClass("FrictionContactConstraint", true);
Creator<Contact::Factory, FrictionContact<FFDDistanceGridCollisionModel, TriangleCollisionModel<sofa::defaulttype::Vec3Types>> > FFDDistanceGridTriangleFrictionContactClass("FrictionContactConstraint", true);


} /// collision
} /// component
} /// sofa
