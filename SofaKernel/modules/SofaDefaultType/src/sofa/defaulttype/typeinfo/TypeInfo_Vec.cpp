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
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>
#include <sofa/defaulttype/TypeInfoRegistryTools.h>
namespace sofa::defaulttype
{

REGISTER_TYPE_INFO_CREATOR(Vec1f);
REGISTER_TYPE_INFO_CREATOR(Vec2f);
REGISTER_TYPE_INFO_CREATOR(Vec3f);
REGISTER_TYPE_INFO_CREATOR(Vec4f);
REGISTER_TYPE_INFO_CREATOR(Vec6f);

REGISTER_TYPE_INFO_CREATOR(Vec1d);
REGISTER_TYPE_INFO_CREATOR(Vec2d);
REGISTER_TYPE_INFO_CREATOR(Vec3d);
REGISTER_TYPE_INFO_CREATOR(Vec4d);
REGISTER_TYPE_INFO_CREATOR(Vec6d);

REGISTER_TYPE_INFO_CREATOR(Vec1i);
REGISTER_TYPE_INFO_CREATOR(Vec2i);
REGISTER_TYPE_INFO_CREATOR(Vec3i);
REGISTER_TYPE_INFO_CREATOR(Vec4i);
REGISTER_TYPE_INFO_CREATOR(Vec6i);

REGISTER_TYPE_INFO_CREATOR(Vec1u);
REGISTER_TYPE_INFO_CREATOR(Vec2u);
REGISTER_TYPE_INFO_CREATOR(Vec3u);
REGISTER_TYPE_INFO_CREATOR(Vec4u);
REGISTER_TYPE_INFO_CREATOR(Vec6u);

} /// namespace sofa::defaulttype

