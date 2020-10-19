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

namespace sofa::defaulttype
{

template<>
struct DataTypeInfo<char> : public IntegerTypeInfo<char>
{
    static const char* name() { return "char"; }
};

template<>
struct DataTypeInfo<unsigned char> : public IntegerTypeInfo<unsigned char>
{
    static const char* name() { return "unsigned char"; }
};

template<>
struct DataTypeInfo<short> : public IntegerTypeInfo<short>
{
    static const char* name() { return "short"; }
};

template<>
struct DataTypeInfo<unsigned short> : public IntegerTypeInfo<unsigned short>
{
    static const char* name() { return "unsigned short"; }
};

template<>
struct DataTypeInfo<int> : public IntegerTypeInfo<int>
{
    static const char* name() { return "int"; }
};

template<>
struct DataTypeInfo<unsigned int> : public IntegerTypeInfo<unsigned int>
{
    static const char* name() { return "unsigned int"; }
};

template<>
struct DataTypeInfo<long> : public IntegerTypeInfo<long>
{
    static const char* name() { return "long"; }
};

template<>
struct DataTypeInfo<unsigned long> : public IntegerTypeInfo<unsigned long>
{
    static const char* name() { return "unsigned long"; }
};

template<>
struct DataTypeInfo<long long> : public IntegerTypeInfo<long long>
{
    static const char* name() { return "long long"; }
};

template<>
struct DataTypeInfo<unsigned long long> : public IntegerTypeInfo<unsigned long long>
{
    static const char* name() { return "unsigned long long"; }
};


} /// namespace sofa::defaulttype

