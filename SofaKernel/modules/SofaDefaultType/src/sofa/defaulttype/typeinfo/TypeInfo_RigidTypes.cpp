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
#include <sofa/defaulttype/typeinfo/TypeInfo_RigidTypes.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>

namespace sofa::defaulttype
{

REGISTER_TYPE_INFO_CREATOR(Rigid2fMass)
REGISTER_TYPE_INFO_CREATOR(Rigid2dMass)
REGISTER_TYPE_INFO_CREATOR(Rigid3fMass)
REGISTER_TYPE_INFO_CREATOR(Rigid3dMass)

REGISTER_TYPE_INFO_CREATOR(Rigid2fTypes)
REGISTER_TYPE_INFO_CREATOR(Rigid2dTypes)
REGISTER_TYPE_INFO_CREATOR(Rigid2fTypes::Coord)
REGISTER_TYPE_INFO_CREATOR(Rigid2dTypes::Coord)
REGISTER_TYPE_INFO_CREATOR(Rigid2fTypes::Deriv)
REGISTER_TYPE_INFO_CREATOR(Rigid2dTypes::Deriv)

REGISTER_TYPE_INFO_CREATOR(Rigid3fTypes)
REGISTER_TYPE_INFO_CREATOR(Rigid3dTypes)
REGISTER_TYPE_INFO_CREATOR(Rigid3fTypes::Coord)
REGISTER_TYPE_INFO_CREATOR(Rigid3dTypes::Coord)
REGISTER_TYPE_INFO_CREATOR(Rigid3fTypes::Deriv)
REGISTER_TYPE_INFO_CREATOR(Rigid3dTypes::Deriv)

} /// namespace sofa::defaulttype

