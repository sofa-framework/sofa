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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Vec.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{

static int Vec1dRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec1d), VirtualTypeInfoA< DataTypeInfo<Vec1d> >::get());
static int Vec2dRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec2d), VirtualTypeInfoA< DataTypeInfo<Vec2d> >::get());
static int Vec3dRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec3d), VirtualTypeInfoA< DataTypeInfo<Vec3d> >::get());
static int Vec4dRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec4d), VirtualTypeInfoA< DataTypeInfo<Vec4d> >::get());
static int Vec6dRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec6d), VirtualTypeInfoA< DataTypeInfo<Vec6d> >::get());

static int Vec1fRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec1f), VirtualTypeInfoA< DataTypeInfo<Vec1f> >::get());
static int Vec2fRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec2f), VirtualTypeInfoA< DataTypeInfo<Vec2f> >::get());
static int Vec3fRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec3f), VirtualTypeInfoA< DataTypeInfo<Vec3f> >::get());
static int Vec4fRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec4f), VirtualTypeInfoA< DataTypeInfo<Vec4f> >::get());
static int Vec6fRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec6f), VirtualTypeInfoA< DataTypeInfo<Vec6f> >::get());

static int Vec1iRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec1i), VirtualTypeInfoA< DataTypeInfo<Vec1i> >::get());
static int Vec2iRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec2i), VirtualTypeInfoA< DataTypeInfo<Vec2i> >::get());
static int Vec3iRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec3i), VirtualTypeInfoA< DataTypeInfo<Vec3i> >::get());
static int Vec4iRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec4i), VirtualTypeInfoA< DataTypeInfo<Vec4i> >::get());
static int Vec6iRegistryIndex = DataTypeInfoRegistry::Set(typeid(Vec6i), VirtualTypeInfoA< DataTypeInfo<Vec6i> >::get());


} /// namespace sofa::defaulttype

