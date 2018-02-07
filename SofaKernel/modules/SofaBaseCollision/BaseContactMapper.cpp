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
#define SOFA_COMPONENT_COLLISION_BASECONTACTMAPPER_CPP

#include <SofaBaseCollision/BaseContactMapper.h>

#include <sofa/helper/Factory.inl>

#include <sofa/defaulttype/VecTypes.h>

namespace sofa
{

namespace component
{

namespace collision
{

using namespace defaulttype;

#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Vec3fTypes>;
template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Rigid3fTypes>;
#endif
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Vec3dTypes>;
template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Rigid3dTypes>;
#endif

} // namespace collision

} // namespace component

namespace helper
{
#ifndef SOFA_DOUBLE
template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Vec3fTypes>, core::CollisionModel* >;
template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Rigid3fTypes>, core::CollisionModel* >;
#endif
#ifndef SOFA_FLOAT
template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Vec3dTypes>, core::CollisionModel* >;
template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Rigid3dTypes>, core::CollisionModel* >;
#endif
} // namespace helper

} // namespace sofa
