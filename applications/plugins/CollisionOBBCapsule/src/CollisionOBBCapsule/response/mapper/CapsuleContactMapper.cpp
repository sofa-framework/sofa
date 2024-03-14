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
#define SOFA_SOFAMISCCOLLISION_CAPSULECONTACTMAPPER_CPP
#include <CollisionOBBCapsule/response/mapper/CapsuleContactMapper.h>

#include <sofa/component/collision/response/mapper/BarycentricContactMapper.inl>
#include <sofa/component/collision/response/mapper/RigidContactMapper.inl>
#include <CollisionOBBCapsule/geometry/CapsuleModel.h>
#include <CollisionOBBCapsule/geometry/RigidCapsuleModel.h>

using namespace sofa::core::collision;
using namespace sofa::component::collision;

namespace sofa::component::collision::response::mapper
{

ContactMapperCreator< ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>> > CapsuleContactMapperClass("PenalityContactForceField", true);
ContactMapperCreator< ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, sofa::defaulttype::Vec3Types> > RigidCapsuleContactMapperClass("PenalityContactForceField", true);
template class COLLISIONOBBCAPSULE_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Vec3Types>, sofa::defaulttype::Vec3Types>;
template class COLLISIONOBBCAPSULE_API ContactMapper<CapsuleCollisionModel<sofa::defaulttype::Rigid3Types>, sofa::defaulttype::Vec3Types>;

} // namespace sofa::component::collision::response::mapper
