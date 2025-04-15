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
#define SOFA_COMPONENT_COLLISION_RIGIDCAPSULECOLLISIONMODEL_CPP
#include <CollisionOBBCapsule/geometry/RigidCapsuleModel.inl>
#include <sofa/core/ObjectFactory.h>

namespace collisionobbcapsule::geometry
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;

void registerRigidCapsuleCollisionModel(sofa::core::ObjectFactory* factory)
{
        factory->registerObjects(sofa::core::ObjectRegistrationData("Collision model which represents a set of rigid capsules.")
        .add<  CapsuleCollisionModel<defaulttype::Rigid3Types> >());
}

template class COLLISIONOBBCAPSULE_API geometry::TCapsule<defaulttype::Rigid3Types>;
template class COLLISIONOBBCAPSULE_API CapsuleCollisionModel<defaulttype::Rigid3Types>;

} // namespace collisionobbcapsule::geometry
