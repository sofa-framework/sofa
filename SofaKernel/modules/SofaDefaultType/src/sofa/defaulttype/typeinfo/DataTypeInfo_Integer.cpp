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
#include <sofa/defaulttype/typeinfo/DataTypeInfo_Integer.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>

namespace sofa::defaulttype
{

static int CharTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(char), VirtualTypeInfoA< DataTypeInfo<char> >::get());
static int UCharTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(unsigned char), VirtualTypeInfoA< DataTypeInfo<unsigned char> >::get());
static int ShortTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(short), VirtualTypeInfoA< DataTypeInfo<short> >::get());
static int UShortTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(unsigned short), VirtualTypeInfoA< DataTypeInfo<unsigned short> >::get());
static int IntTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(int), VirtualTypeInfoA< DataTypeInfo<int> >::get());
static int UIntTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(unsigned int), VirtualTypeInfoA< DataTypeInfo<unsigned int> >::get());
static int LongTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(long), VirtualTypeInfoA< DataTypeInfo<long> >::get());
static int ULongTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(unsigned long), VirtualTypeInfoA< DataTypeInfo<unsigned long> >::get());
static int LongLongTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(long long), VirtualTypeInfoA< DataTypeInfo<long long> >::get());
static int ULongLongTypeInfo = DataTypeInfoRegistry::RegisterTypeInfo(typeid(unsigned long long), VirtualTypeInfoA< DataTypeInfo<unsigned long long> >::get());

} /// namespace sofa::defaulttype

