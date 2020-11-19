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
#include <CImgPlugin/CImgData.h>

namespace sofa::defaulttype
{

////// infos for Data
class BaseImageTypeInfo
{
public:
    virtual ~BaseImageTypeInfo(){}
};

template<class TDataType>
struct ImageTypeInfo : public BaseImageTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::T BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       }; ///< 1 if this type has valid infos
    enum { FixedSize       = 1                             }; ///< 1 if this type has a fixed size  -> always 1 Image
    enum { ZeroConstructor = 0                             }; ///< 1 if the constructor is equivalent to setting memory to 0  -> I guess so, a default Image is initialzed with nothing
    enum { SimpleCopy      = 0                             }; ///< 1 if copying the data can be done with a memcpy
    enum { SimpleLayout    = 0                             }; ///< 1 if the layout in memory is simply N values of the same base type
    enum { Integer         = 0                             }; ///< 1 if this type uses integer values
    enum { Scalar          = 0                             }; ///< 1 if this type uses scalar values
    enum { Text            = 0                             }; ///< 1 if this type uses text values
    enum { CopyOnWrite     = 1                             }; ///< 1 if this type uses copy-on-write -> it seems to be THE important option not to perform too many copies
    enum { Container       = 0                             }; ///< 1 if this type is a container

    enum { Size = 1 }; ///< largest known fixed size for this type, as returned by size()

    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return 1; }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &/*data*/, Index /*index*/, T& /*value*/)
    {
        return;
    }

    template<typename T>
    static void setValue(DataType &/*data*/, Index /*index*/, const T& /*value*/ )
    {
        return;
    }

    static void getValueString(const DataType &data, Index index, std::string& value)
    {
        if (index != 0) return;
        std::ostringstream o; o << data; value = o.str();
    }

    static void setValueString(DataType &data, Index index, const std::string& value )
    {
        if (index != 0) return;
        std::istringstream i(value); i >> data;
    }

    static const void* getValuePtr(const DataType&)
    {
        return nullptr;
    }

    static void* getValuePtr(DataType&)
    {
        return nullptr;
    }
};


template<class T>
struct DataTypeInfo< Image<T> > : public ImageTypeInfo< Image<T> >
{
    static std::string name() { std::ostringstream o; o << "Image<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

} /// namespace

