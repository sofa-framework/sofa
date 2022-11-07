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

#include <sofa/type/BoundingBox.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Scalar.h>
#include <sofa/defaulttype/typeinfo/TypeInfo_Vec.h>


namespace sofa::defaulttype
{

struct BoundingBoxTypeInfo
{
    typedef type::BoundingBox DataType;
    typedef type::Vec3 BaseType;
    typedef DataTypeInfo<type::Vec3d> BaseTypeInfo;
    typedef SReal ValueType;

    enum
    {
        ValidInfo = BaseTypeInfo::ValidInfo
    };  ///< 1 if this type has valid infos
    enum
    {
        FixedSize = 1
    };  ///< 1 if this type has a fixed size  -> always 1 single pair of vec3
    enum
    {
        ZeroConstructor = 0
    };  ///< 1 if the constructor is equivalent to setting memory to 0
    /// -> I don't think so, bbox are initialized with +inf / -inf usually
    enum
    {
        SimpleCopy = 1
    };  ///< 1 if copying the data can be done with a memcpy
    enum
    {
        SimpleLayout = 1
    };  ///< 1 if the layout in memory is simply N values of the same base type
    enum
    {
        Integer = 0
    };  ///< 1 if this type uses integer values
    enum
    {
        Scalar = 1
    };  ///< 1 if this type uses scalar values
    enum
    {
        Text = 0
    };  ///< 1 if this type uses text values
    enum
    {
        CopyOnWrite = 1
    };  ///< 1 if this type uses copy-on-write -> it seems to be THE important
    ///< option not to perform too many copiesf
    enum
    {
        Container = 1
    };  ///< 1 if this type is a container

    enum
    {
        Size = 2
    };  ///< largest known fixed size for this type, as returned by size()

    static sofa::Size size() { return 3; } // supposed to be the total number of elements. Ends up being the number of elements in the 2nd dimension
    static sofa::Size byteSize() { return sizeof (ValueType); }  // Size of the smalest single element in the container: BoundingBox uses Vec3d internally, so double

    static sofa::Size size(const DataType & /*data*/) { return 2 * BaseTypeInfo::size(); } // supposed to be the nb of elements in the 1st dimension. Ends up being the total number of elems.

    static bool setSize(DataType & /*data*/, sofa::Size /*size*/) { return false; } // FixedArray -> ignore

    template <typename T>
    static void getValue(const DataType & data, sofa::Size index,
                         T & value) /// since TypeInfos abstract all containers as 1D arrays, T here is of ValueType
    {
        value = static_cast<T>(((ValueType*)&data)[index]);
    }

    template <typename T>
    static void setValue(DataType & data, sofa::Size index,
                         const T & value)
    {
        ((ValueType*)&data)[index] = static_cast<ValueType>(value);
    }

    static double getScalarValue (const void* data, sofa::Size index)
    {
        return ((ValueType*)&data)[index];
    }

    static void setScalarValue (const void* data, sofa::Size index, SReal value)
    {
        ((ValueType*)&data)[index] = value;
    }


    static void getValueString(const DataType &data, sofa::Size index,
                               std::string &value)
    {
        if (index != 0) return;
        std::ostringstream o;
        o << data;
        value = o.str();
    }

    static void setValueString(DataType &data, sofa::Size index,
                               const std::string &value)
    {
        if (index != 0) return;
        std::istringstream i(value);
        i >> data;
    }

    static const void *getValuePtr(const DataType & bbox) { return (const void*)(&bbox); }

    static void *getValuePtr(DataType &bbox) { return (void*)(&bbox); }
};

template <>
struct DataTypeInfo<type::BoundingBox> : public BoundingBoxTypeInfo
{
    static std::string name() { return "BoundingBox"; }
    static std::string GetTypeName() { return "BoundingBox"; }
};
} /// namespace sofa::defaulttype

