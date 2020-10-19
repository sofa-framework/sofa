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
struct SetTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::value_type BaseType;
    typedef DataTypeInfo<BaseType> BaseTypeInfo;
    typedef typename BaseTypeInfo::ValueType ValueType;
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    enum { ValidInfo       = BaseTypeInfo::ValidInfo       };
    enum { FixedSize       = 0                             };
    enum { ZeroConstructor = 0                             };
    enum { SimpleCopy      = 0                             };
    enum { SimpleLayout    = 0                             };
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
            size_t s = 0;
            for (typename DataType::const_iterator it = data.begin(), end=data.end(); it!=end; ++it)
                s+= BaseTypeInfo::size(*it);
            return s;
        }
    }

    static bool setSize(DataType& data, size_t /*size*/)
    {
        data.clear(); // we can't "resize" a set, so the only meaningfull operation is to clear it, as values will be added dynamically in setValue
        return true;
    }

    template <typename T>
    static void getValue(const DataType &data, size_t index, T& value)
    {
        if (BaseTypeInfo::FixedSize)
        {
            typename DataType::const_iterator it = data.begin();
            for (size_t i=0; i<index/BaseTypeInfo::size(); ++i) ++it;
            BaseTypeInfo::getValue(*it, index%BaseTypeInfo::size(), value);
        }
        else
        {
            size_t s = 0;
            for (typename DataType::const_iterator it = data.begin(), end=data.end(); it!=end; ++it)
            {
                size_t n = BaseTypeInfo::size(*it);
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(*it, index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &data, size_t /*index*/, const T& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseType t;
            BaseTypeInfo::setValue(t, 0, value);
            data.insert(t);
        }
        else
        {
            msg_error("SetTypeInfo") << "setValue not implemented for set with composite values.";
        }
    }

    static void getValueString(const DataType &data, size_t index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize)
        {
            typename DataType::const_iterator it = data.begin();
            for (size_t i=0; i<index/BaseTypeInfo::size(); ++i) ++it;
            BaseTypeInfo::getValueString(*it, index%BaseTypeInfo::size(), value);
        }
        else
        {
            size_t s = 0;
            for (typename DataType::const_iterator it = data.begin(), end=data.end(); it!=end; ++it)
            {
                size_t n = BaseTypeInfo::size(*it);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(*it, index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &data, size_t /*index*/, const std::string& value )
    {
        if (BaseTypeInfo::FixedSize && BaseTypeInfo::size() == 1)
        {
            BaseType t;
            BaseTypeInfo::setValueString(t, 0, value);
            data.insert(t);
        }
        else
        {
            msg_error("SetTypeInfo") << "setValueString not implemented for set with composite values.";
        }
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

