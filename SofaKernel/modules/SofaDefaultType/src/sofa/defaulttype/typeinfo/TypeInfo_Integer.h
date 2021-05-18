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
    static const std::string GetTypeName() { return "char"; }
    static const std::string name() { return "b"; }
};

template<>
struct DataTypeInfo<unsigned char> : public IntegerTypeInfo<unsigned char>
{
    static const std::string GetTypeName() { return "unsigned char"; }
    static const std::string name() { return "B"; }
};

template<>
struct DataTypeInfo<short> : public IntegerTypeInfo<short>
{
    static const std::string GetTypeName() { return "short"; }
    static const std::string name() { return "h"; }
};

template<>
struct DataTypeInfo<unsigned short> : public IntegerTypeInfo<unsigned short>
{
    static const std::string GetTypeName() { return "unsigned short"; }
    static const std::string name() { return "H"; }
};

template<>
struct DataTypeInfo<int> : public IntegerTypeInfo<int>
{
    static const std::string GetTypeName() { return "int"; }
    static const std::string name() { return "i"; }
};

template<>
struct DataTypeInfo<unsigned int> : public IntegerTypeInfo<unsigned int>
{
    static const std::string GetTypeName() { return "unsigned int"; }
    static const std::string name() { return "I"; }
};

template<>
struct DataTypeInfo<long> : public IntegerTypeInfo<long>
{
    static const std::string GetTypeName() { return "long"; }
    static const std::string name() { return "l"; }
};

template<>
struct DataTypeInfo<unsigned long> : public IntegerTypeInfo<unsigned long>
{
    static const std::string GetTypeName() { return "unsigned long"; }
    static const std::string name() { return "L"; }
};

template<>
struct DataTypeInfo<long long> : public IntegerTypeInfo<long long>
{
    static const std::string GetTypeName() { return "long long"; }
    static const std::string name() { return "q"; }
};

template<>
struct DataTypeInfo<unsigned long long> : public IntegerTypeInfo<unsigned long long>
{
    static const std::string GetTypeName() { return "unsigned long long"; }
    static const std::string name() { return "Q"; }
};


} /// typeNamespace sofa::defaulttype

