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

#include <sstream>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>

namespace sofa::defaulttype
{

template<class TDataType>
struct ScalarTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType;
    typedef ScalarTypeInfo<TDataType> BaseTypeInfo;
    typedef ScalarTypeInfo<TDataType> ValueTypeInfo;

    enum { ValidInfo       = 1 };
    enum { FixedSize       = 1 };
    enum { ZeroConstructor = 1 };
    enum { SimpleCopy      = 1 };
    enum { SimpleLayout    = 1 };
    enum { Integer         = 0 };
    enum { Scalar          = 1 };
    enum { Text            = 0 };
    enum { CopyOnWrite     = 0 };
    enum { Container       = 0 };

    enum { Size = 1 };
    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return sizeof(DataType); }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &data, sofa::Size index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(data);
    }

    template<typename T>
    static void setValue(DataType &data, sofa::Size index, const T& value )
    {
        if (index != 0) return;
        data = static_cast<DataType>(value);
    }

    static void getValueString(const DataType &data, sofa::Size index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << data; value = o.str();
    }

    static void setValueString(DataType &data, sofa::Size index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> data;
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

