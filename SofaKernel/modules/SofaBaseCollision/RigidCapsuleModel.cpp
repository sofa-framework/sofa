/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_COLLISION_RIGIDCAPSULEMODEL_CPP
#include "RigidCapsuleModel.inl"

namespace sofa
{

namespace component
{

namespace collision
{

using namespace sofa::defaulttype;
using namespace sofa::core::collision;
using namespace helper;

SOFA_DECL_CLASS(RigidCapsule)

int RigidCapsuleModelClass = core::RegisterObject("Collision model which represents a set of rigid capsules")
#ifndef SOFA_FLOAT
        .add<  TCapsuleModel<defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add < TCapsuleModel<defaulttype::Rigid3fTypes> >()
#endif
        .addAlias("RigidCapsule")
        .addAlias("RigidCapsuleModel")
//.addAlias("CapsuleMesh")
//.addAlias("CapsuleSet")
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API TCapsule<defaulttype::Rigid3dTypes>;
template class SOFA_BASE_COLLISION_API TCapsuleModel<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API TCapsule<defaulttype::Rigid3fTypes>;
template class SOFA_BASE_COLLISION_API TCapsuleModel<defaulttype::Rigid3fTypes>;
#endif



}
}
}
