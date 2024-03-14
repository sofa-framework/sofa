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

#include <sofa/defaulttype/config.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>

namespace sofa::defaulttype
{
template<class TDataType>
struct IncompleteTypeInfo
{
    /// Template parameter.
    typedef TDataType DataType;
    /// If the type is a container, this the type of the values inside this
    /// container, otherwise this is DataType.
    typedef DataType BaseType;
    /// Type of the final atomic values (i.e. the values indexed by getValue()).
    typedef DataType ValueType;
    /// TypeInfo for BaseType
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    /// TypeInfo for ValueType
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = 0 /**< 1 if this type has valid infos*/ };
    enum { FixedSize       = 0 /**< 1 if this type has a fixed size*/ };
    enum { ZeroConstructor =  0 /**< 1 if the constructor is equivalent to setting memory to 0*/ };
    enum { SimpleCopy      = 0 /**< 1 if copying the data can be done with a memcpy*/ };
    enum { SimpleLayout    = 0 /**< 1 if the layout in memory is simply N values of the same base type*/ };
    enum { Integer         = 0 /**< 1 if this type uses integer values*/ };
    enum { Scalar          = 0 /**< 1 if this type uses scalar values*/ };
    enum { Text            = 0 /**< 1 if this type uses text values*/ };
    enum { CopyOnWrite     = 0 /**< 1 if this type uses copy-on-write. The memory is shared with its source Data while only the source is changing (and the source modifications are then visible in the current Data). As soon as modifications are applied to the current Data, it will allocate its own value, and no longer shares memory with the source.*/ };
    enum { Container       = 0 /**< 1 if this type is a container*/ };
    enum { Size = 0 /**< largest known fixed size for this type, as returned by size() */ };

    static sofa::Size size() { return 0; }
    static sofa::Size byteSize() { return 0; }
    static sofa::Size size(const DataType& /*data*/) { return 1; }

    template <typename T>
    static void getValue(const DataType& /*data*/, sofa::Size /*index*/, T& /*value*/)
    {}

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template<typename T>
    static void setValue(DataType& /*data*/, sofa::Size /*index*/, const T& /*value*/){}

    static void getValueString(const DataType& /*data*/, sofa::Size /*index*/, std::string& /*value*/){}

    static const void* getValuePtr(const TDataType& data)
    {
        SOFA_UNUSED(data);
        return nullptr;
    }

    static void* getValuePtr(TDataType& data)
    {
        SOFA_UNUSED(data);
        return nullptr;
    }

    static void setValueString(DataType &data, sofa::Size index, const std::string& value)
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        SOFA_UNUSED(value);
    }

    static const std::string name() { return GetTypeName(); }
    static const std::string GetTypeName() { return sofa::helper::NameDecoder::decodeFullName(typeid(DataType)); }
};


} /// namespace sofa::defaulttype

