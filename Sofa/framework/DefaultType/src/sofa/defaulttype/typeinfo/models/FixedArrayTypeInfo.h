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

#include <sofa/type/fixed_array.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>

namespace sofa::defaulttype
{

template<class TDataType, sofa::Size static_size = TDataType::static_size>
struct FixedArrayTypeInfo
{
    typedef TDataType DataType;

    typedef typename DataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       };
    enum { FixedSize       = BaseTypeInfo::FixedSize       };
    enum { ZeroConstructor = BaseTypeInfo::ZeroConstructor };
    enum { SimpleCopy      = BaseTypeInfo::SimpleCopy      };
    enum { SimpleLayout    = BaseTypeInfo::SimpleLayout    };
    enum { Integer         = BaseTypeInfo::Integer         };
    enum { Scalar          = BaseTypeInfo::Scalar          };
    enum { Text            = BaseTypeInfo::Text            };
    enum { CopyOnWrite     = 1                             };
    enum { Container       = 1                             };

    enum { Size = static_size * BaseTypeInfo::Size };
    static sofa::Size size()
    {
        return DataType::size() * BaseTypeInfo::size();
    }

    static sofa::Size byteSize()
    {
        return ValueTypeInfo::byteSize();
    }

    static sofa::Size size(const DataType& data)
    {
        if (FixedSize)
            return size();
        else
        {
            sofa::Size s = 0;
            for (sofa::Size i=0; i<DataType::size(); ++i)
                s+= BaseTypeInfo::size(data[(sofa::Size)i]);
            return s;
        }
    }

    static bool setSize(DataType& data, sofa::Size size)
    {
        if (!FixedSize)
        {
            size /= DataType::size();
            for (sofa::Size i=0; i<DataType::size(); ++i)
                if( !BaseTypeInfo::setSize(data[(sofa::Size)i], size) ) return false;
            return true;
        }
        return false;
    }

    template <typename T>
    static void getValue(const DataType &data, sofa::Size index, T& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValue(data[(sofa::Size)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValue(data[(sofa::Size)(index/BaseTypeInfo::size())], (sofa::Size)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            sofa::Size s = 0;
            for (sofa::Size i=0; i<DataType::size(); ++i)
            {
                const sofa::Size n = BaseTypeInfo::size(data[(sofa::Size)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(data[(sofa::Size)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &data, sofa::Size index, const T& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValue(data[(sofa::Size)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValue(data[(sofa::Size)(index/BaseTypeInfo::size())], (sofa::Size)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            sofa::Size s = 0;
            for (sofa::Size i=0; i<DataType::size(); ++i)
            {
                const sofa::Size n = BaseTypeInfo::size(data[(sofa::Size)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValue(data[(sofa::Size)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void getValueString(const DataType &data, sofa::Size index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::getValueString(data[(sofa::Size)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::getValueString(data[(sofa::Size)(index/BaseTypeInfo::size())], (sofa::Size)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            sofa::Size s = 0;
            for (sofa::Size i=0; i<DataType::size(); ++i)
            {
                const sofa::Size n = BaseTypeInfo::size(data[(sofa::Size)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(data[(sofa::Size)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &data, sofa::Size index, const std::string& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseTypeInfo::setValueString(data[(sofa::Size)index], 0, value);
        }
        else if (BaseTypeInfo::FixedSize)
        {
            BaseTypeInfo::setValueString(data[(sofa::Size)(index/BaseTypeInfo::size())], (sofa::Size)(index%BaseTypeInfo::size()), value);
        }
        else
        {
            sofa::Size s = 0;
            for (sofa::Size i=0; i<DataType::size(); ++i)
            {
                const sofa::Size n = BaseTypeInfo::size(data[(sofa::Size)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValueString(data[(sofa::Size)i], index-s, value);
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

