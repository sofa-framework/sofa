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
struct VectorTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::size_type size_type;
    typedef typename DataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       };
    enum { FixedSize       = 0                             };
    enum { ZeroConstructor = 0                             };
    enum { SimpleCopy      = 0                             };
    enum { SimpleLayout    = BaseTypeInfo::SimpleLayout    };
    enum { Integer         = BaseTypeInfo::Integer         };
    enum { Scalar          = BaseTypeInfo::Scalar          };
    enum { Text            = BaseTypeInfo::Text            };
    enum { CopyOnWrite     = 1                             };
    enum { Container       = 1                             };

    enum { Size = BaseTypeInfo::Size };
    static size_t size()
    {
        return BaseTypeInfo::size();
    }

    static size_t byteSize()
    {
        return ValueTypeInfo::byteSize();
    }

    static size_t size(const DataType& data)
    {
        if (BaseTypeInfo::FixedSize)
            return data.size()*BaseTypeInfo::size();
        else
        {
            size_t n = data.size();
            size_t s = 0;
            for (size_t i=0; i<n; ++i)
                s+= BaseTypeInfo::size(data[(size_type)i]);
            return s;
        }
    }

    static bool setSize(DataType& data, size_t size)
    {
        if (BaseTypeInfo::FixedSize)
        {
            data.resize(size/BaseTypeInfo::size());
            return true;
        }
        return false;
    }

    template <typename T>
    static void getValue(const DataType &data, size_t index, T& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValue(data[(size_type)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValue(data[(size_type)(index/BaseTypeInfo::size())], (size_type)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            size_t s = 0;
            for (size_t i=0; i<data.size(); ++i)
            {
                size_t n = BaseTypeInfo::size(data[(size_type)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(data[(size_type)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &data, size_t index, const T& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValue(data[(size_type)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValue(data[(size_type)(index/BaseTypeInfo::size())], (size_type)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            size_t s = 0;
            for (size_t i=0; i<data.size(); ++i)
            {
                size_t n = BaseTypeInfo::size(data[(size_type)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValue(data[(size_type)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void getValueString(const DataType &data, size_t index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValueString(data[(size_type)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValueString(data[(size_type)(index/BaseTypeInfo::size())], (size_type)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            size_t s = 0;
            for (size_t i=0; i<data.size(); ++i)
            {
                size_t n = BaseTypeInfo::size(data[(size_type)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(data[(size_type)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &data, size_t index, const std::string& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValueString(data[(size_type)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValueString(data[(size_type)(index/BaseTypeInfo::size())], (size_type)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            size_t s = 0;
            for (size_t i=0; i<data.size(); ++i)
            {
                size_t n = BaseTypeInfo::size(data[(size_type)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValueString(data[(size_type)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static const void* getValuePtr(const DataType& data)
    {
        return &data[0];
    }

    static void* getValuePtr(DataType& data)
    {
        return &data[0];
    }
};



} /// namespace sofa::defaulttype

