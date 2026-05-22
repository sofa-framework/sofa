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
#pragma once

#include <sofa/defaulttype/typeinfo/models/IntegerTypeInfo.h>
#include <sofa/type/trait/TypeTrait.h>

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo<char> : public IntegerTypeInfo<char>, public type::TypeTrait<char>
{};

template<>
struct DataTypeInfo<unsigned char> : public IntegerTypeInfo<unsigned char>, public sofa::type::TypeTrait<unsigned char>
{};

template<>
struct DataTypeInfo<short> : public IntegerTypeInfo<short>, public sofa::type::TypeTrait<short>
{};

template<>
struct DataTypeInfo<unsigned short> : public IntegerTypeInfo<unsigned short>, public sofa::type::TypeTrait<unsigned short>
{};

template<>
struct DataTypeInfo<int> : public IntegerTypeInfo<int>, public sofa::type::TypeTrait<int>
{};

template<>
struct DataTypeInfo<unsigned int> : public IntegerTypeInfo<unsigned int>, public sofa::type::TypeTrait<unsigned int>
{};

template<>
struct DataTypeInfo<long> : public IntegerTypeInfo<long>, public sofa::type::TypeTrait<long>
{};

template<>
struct DataTypeInfo<unsigned long> : public IntegerTypeInfo<unsigned long>, public sofa::type::TypeTrait<unsigned long>
{};

template<>
struct DataTypeInfo<long long> : public IntegerTypeInfo<long long>, public sofa::type::TypeTrait<long long>
{};

template<>
struct DataTypeInfo<unsigned long long> : public IntegerTypeInfo<unsigned long long>, public sofa::type::TypeTrait<unsigned long long>
{};

} /// typeNamespace sofa::defaulttype

