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

template<class TDataType>
struct TextTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType;
    typedef TextTypeInfo<TDataType> BaseTypeInfo;
    typedef TextTypeInfo<TDataType> ValueTypeInfo;

    enum { ValidInfo       = 1 };
    enum { FixedSize       = 0 };
    enum { ZeroConstructor = 0 };
    enum { SimpleCopy      = 0 };
    enum { SimpleLayout    = 0 };
    enum { Integer         = 0 };
    enum { Scalar          = 0 };
    enum { Text            = 1 };
    enum { CopyOnWrite     = 1 };
    enum { Container       = 0 };

    enum { Size = 1 };
    static size_t size() { return 1; }
    static size_t byteSize() { return 1; }

    static size_t size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, size_t /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &data, size_t index, T& value)
    {
        if (index != 0) return;
        std::istringstream i(data); i >> value;
    }

    template<typename T>
    static void setValue(DataType &data, size_t index, const T& value )
    {
        if (index != 0) return;
        std::ostringstream o; o << value; data = o.str();
    }

    static void getValueString(const DataType &data, size_t index, std::string& value)
    {
        if (index != 0) return;
        value = data;
    }

    static void setValueString(DataType &data, size_t index, const std::string& value )
    {
        if (index != 0) return;
        data = value;
    }

    static const void* getValuePtr(const DataType& /*data*/)
    {
        return nullptr;
    }

    static void* getValuePtr(DataType& /*data*/)
    {
        return nullptr;
    }
};

} /// namespace sofa::defaulttype

