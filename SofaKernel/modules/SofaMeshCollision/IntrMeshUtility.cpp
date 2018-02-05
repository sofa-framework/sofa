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
#define SOFA_COMPONENT_COLLISION_INTRMESHUTILITY_CPP
#include <SofaMeshCollision/IntrMeshUtility.inl>

namespace sofa{
namespace component{
namespace collision{

#ifndef SOFA_FLOAT
template struct SOFA_MESH_COLLISION_API IntrUtil<TTriangle<defaulttype::Vec3dTypes> >;
template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<defaulttype::Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<defaulttype::Vec3dTypes>,TOBB<defaulttype::Rigid3dTypes> >;
template struct SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<defaulttype::Vec3dTypes> >;
#endif
#ifndef SOFA_DOUBLE
template struct SOFA_MESH_COLLISION_API IntrUtil<TTriangle<defaulttype::Vec3fTypes> >;
template class SOFA_MESH_COLLISION_API FindContactSet<TTriangle<defaulttype::Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
template class SOFA_MESH_COLLISION_API IntrAxis<TTriangle<defaulttype::Vec3fTypes>,TOBB<defaulttype::Rigid3fTypes> >;
template struct SOFA_MESH_COLLISION_API IntrConfigManager<TTriangle<defaulttype::Vec3fTypes> >;
#endif


}
}
}
