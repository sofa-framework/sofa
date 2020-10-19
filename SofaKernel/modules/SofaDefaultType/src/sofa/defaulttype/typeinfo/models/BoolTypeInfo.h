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

#include <sofa/defaulttype/DataTypeInfo.h>

namespace sofa::defaulttype
{

struct BoolTypeInfo
{
    typedef bool DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType;
    typedef BoolTypeInfo BaseTypeInfo;
    typedef BoolTypeInfo ValueTypeInfo;

    enum { ValidInfo       = 1 };
    enum { FixedSize       = 1 };
    enum { ZeroConstructor = 1 };
    enum { SimpleCopy      = 1 };
    enum { SimpleLayout    = 1 };
    enum { Integer         = 1 };
    enum { Scalar          = 0 };
    enum { Text            = 0 };
    enum { CopyOnWrite     = 0 };
    enum { Container       = 0 };

    enum { Size = 1 };
    static size_t size() { return 1; }
    static size_t byteSize() { return sizeof(DataType); }

    static size_t size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, size_t /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &data, size_t index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(data);
    }

    template<typename T>
    static void setValue(DataType &data, size_t index, const T& value )
    {
        if (index != 0) return;
        data = (value != 0);
    }

    template<typename T>
    static void setValue(std::vector<DataType>::reference data, size_t index, const T& v )
    {
        if (index != 0) return;
        data = (v != 0);
    }

    static void getValueString(const DataType &data, size_t index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << data; value = o.str();
    }

    static void setValueString(DataType &data, size_t index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> data;
    }

    static void setValueString(std::vector<DataType>::reference data, size_t index, const std::string& value )
    {
        if (index != 0) return;
        bool b = data;
        std::istringstream i(value); i >> b;
        data = b;
    }

    static const void* getValuePtr(const DataType& data)
    {
        return &data;
    }

    static void* getValuePtr(DataType& data)
    {
        return &data;
    }
};


} /// namespace sofa::defaulttype

