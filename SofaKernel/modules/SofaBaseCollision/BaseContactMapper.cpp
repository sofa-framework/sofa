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

std::string GenerateStringID::generate(){
    static std::string alphanum = "0123456789!@#$%^&*ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    std::string result;
    result.resize(length);
    for (int i = 0; i < length; i++)
        result[i] = alphanum[rand() % length];

    return result;
}



template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Vec3Types>;
template class SOFA_BASE_COLLISION_API BaseContactMapper<defaulttype::Rigid3Types>;


} // namespace collision

} // namespace component

namespace helper
{
template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Vec3Types>, core::CollisionModel* >;
template class SOFA_BASE_COLLISION_API Factory< std::string, sofa::component::collision::BaseContactMapper<defaulttype::Rigid3Types>, core::CollisionModel* >;

} // namespace helper

} // namespace sofa
