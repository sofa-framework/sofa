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
#ifndef SOFA_DEFAULTTYPE_DATATYPEINFO_H
#define SOFA_DEFAULTTYPE_DATATYPEINFO_H

#include <vector>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/set.h>
#include <sofa/helper/types/RGBAColor.h>
#include <sstream>
#include <typeinfo>
#include <sofa/helper/logging/Messaging.h>
#include "AbstractTypeInfo.h"

namespace sofa
{

namespace helper
{
template <class T, class MemoryManager >
class vector;
}

namespace defaulttype
{

/** Type traits class for objects stored in Data.

    %DataTypeInfo is part of the introspection/reflection capabilities of the
    Sofa scene graph API; it is used to manipulate Data values generically in \a
    template code, working transparently with different types of containers
    (vector, fixed_array, etc), and different types of values (integers, scalars
    (float, double), strings, etc). For example, it can be used to work with
    arrays without having to handle all the possible array classes used in Sofa:
    fixed or dynamic size, CPU or GPU, etc.

    <h4>Small example</h4>

    Iterate over the values of a DataType in templated code:

    \code{.cpp}
    template<DataType>
    MyComponent::someMethod(DataType& data) {
        const size_t dim = defaulttype::DataTypeInfo<Coord>::size();
        for(size_t i = 0; i < dim; ++i) {
            DataTypeInfo<DataType>::ValueType value;
            DataTypeInfo<Coord>::getValue(data, i, value);
            // [...] Do something with 'value'
        }
    }
    \endcode


    <h4>Note about size and indices</h4>

    The getValue() and setValue() methods take an index as a parameter, with the
    following conventions:

    - If a type is not a container, then the index \b must be 0.

    - Multi-dimensional containers are abstracted to a single dimension.  This
      allows iterating over any container using a single index, at the price of
      some limitations.

    \see AbstractTypeInfo provides similar mechanisms to manipulate Data objects
    generically in non-template code.
*/
template<class TDataType>
struct DataTypeInfo;

template<class TDataType>
struct DefaultDataTypeInfo
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
    /**
       \{
     */
    enum { ValidInfo       = 0 /**< 1 if this type has valid infos*/ };
    enum { FixedSize       = 0 /**< 1 if this type has a fixed size*/ };
    enum { ZeroConstructor = 0 /**< 1 if the constructor is equivalent to setting memory to 0*/ };
    enum { SimpleCopy      = 0 /**< 1 if copying the data can be done with a memcpy*/ };
    enum { SimpleLayout    = 0 /**< 1 if the layout in memory is simply N values of the same base type*/ };
    enum { Integer         = 0 /**< 1 if this type uses integer values*/ };
    enum { Scalar          = 0 /**< 1 if this type uses scalar values*/ };
    enum { Text            = 0 /**< 1 if this type uses text values*/ };
    enum { CopyOnWrite     = 0 /**< 1 if this type uses copy-on-write. The memory is shared with its source Data while only the source is changing (and the source modifications are then visible in the current Data). As soon as modifications are applied to the current Data, it will allocate its own value, and no longer shares memory with the source.*/ };
    enum { Container       = 0 /**< 1 if this type is a container*/ };
    enum { Size            = 1 /**< largest known fixed size for this type, as returned by size() */ };

    // \}

    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return 1; }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    template <typename T>
    static void getValue(const DataType& /*data*/, Index /*index*/, T& /*value*/)
    {
    }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template<typename T>
    static void setValue(DataType& /*data*/, Index /*index*/, const T& /*value*/)
    {
    }

    static void getValueString(const DataType& /*data*/, Index /*index*/, std::string& /*value*/)
    {
    }

    static void setValueString(DataType& /*data*/, Index /*index*/, const std::string& /*value*/)
    {
    }

    // mtournier: wtf is this supposed to do?
    // mtournier: wtf is this not returning &type?
    static const void* getValuePtr(const DataType& /*type*/)
    {
        return nullptr;
    }

    static void* getValuePtr(DataType& /*type*/)
    {
        return nullptr;
    }

    static const char* name() { return "unknown"; }

};

template<class TDataType>
struct DataTypeInfo : DefaultDataTypeInfo<TDataType> { };



/// Type name template: default to using DataTypeInfo::name(), but can be overriden for types with shorter typedefs
template<class TDataType>
struct DataTypeName : public DataTypeInfo<TDataType>
{
};

/// Abstract type traits class
template<class TDataType>
class VirtualTypeInfo : public AbstractTypeInfo
{
public:
    typedef TDataType DataType;
    typedef DataTypeInfo<DataType> Info;

    static VirtualTypeInfo* get() { static VirtualTypeInfo<DataType> t; return &t; }

    const AbstractTypeInfo* BaseType() const override  { return VirtualTypeInfo<typename Info::BaseType>::get(); }
    const AbstractTypeInfo* ValueType() const override { return VirtualTypeInfo<typename Info::ValueType>::get(); }

    virtual std::string name() const override { return DataTypeName<DataType>::name(); }

    bool ValidInfo() const override       { return Info::ValidInfo; }
    bool FixedSize() const override       { return Info::FixedSize; }
    bool ZeroConstructor() const override { return Info::ZeroConstructor; }
    bool SimpleCopy() const override      { return Info::SimpleCopy; }
    bool SimpleLayout() const override    { return Info::SimpleLayout; }
    bool Integer() const override         { return Info::Integer; }
    bool Scalar() const override          { return Info::Scalar; }
    bool Text() const override            { return Info::Text; }
    bool CopyOnWrite() const override     { return Info::CopyOnWrite; }
    bool Container() const override       { return Info::Container; }

    sofa::Size size() const override
    {
        return Info::size();
    }
    sofa::Size byteSize() const override
    {
        return Info::byteSize();
    }
    sofa::Size size(const void* data) const override
    {
        return sofa::Size(Info::size(*(const DataType*)data));
    }
    bool setSize(void* data, sofa::Size size) const override
    {
        return Info::setSize(*(DataType*)data, size);
    }

    long long getIntegerValue(const void* data, Index index) const override
    {
        long long v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    double    getScalarValue (const void* data, Index index) const override
    {
        double v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    virtual std::string getTextValue   (const void* data, Index index) const override
    {
        std::string v;
        Info::getValueString(*(const DataType*)data, index, v);
        return v;
    }

    void setIntegerValue(void* data, Index index, long long value) const override
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    void setScalarValue (void* data, Index index, double value) const override
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    virtual void setTextValue(void* data, Index index, const std::string& value) const override
    {
        Info::setValueString(*(DataType*)data, index, value);
    }
    const void* getValuePtr(const void* data) const override
    {
        return Info::getValuePtr(*(const DataType*)data);
    }
    void* getValuePtr(void* data) const override
    {
        return Info::getValuePtr(*(DataType*)data);
    }

    virtual const std::type_info* type_info() const override { return &typeid(DataType); }


protected: // only derived types can instantiate this class
    VirtualTypeInfo() {}
};

template<class TDataType>
struct IntegerTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType;
    typedef IntegerTypeInfo<DataType> BaseTypeInfo;
    typedef IntegerTypeInfo<DataType> ValueTypeInfo;

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
    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return sizeof(DataType); }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &data, Index index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(data);
    }

    template<typename T>
    static void setValue(DataType &data, Index index, const T& value )
    {
        if (index != 0) return;
        data = static_cast<DataType>(value);
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

    static const void* getValuePtr(const DataType& data)
    {
        return &data;
    }

    static void* getValuePtr(DataType& data)
    {
        return &data;
    }
};

struct BoolTypeInfo
{
    typedef bool DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType;
    typedef IntegerTypeInfo<DataType> BaseTypeInfo;
    typedef IntegerTypeInfo<DataType> ValueTypeInfo;

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
    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return sizeof(DataType); }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &data, Index index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(data);
    }

    template<typename T>
    static void setValue(DataType &data, Index index, const T& value )
    {
        if (index != 0) return;
        data = (value != 0);
    }

    template<typename T>
    static void setValue(std::vector<DataType>::reference data, Index index, const T& v )
    {
        if (index != 0) return;
        data = (v != 0);
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

    static void setValueString(std::vector<DataType>::reference data, Index index, const std::string& value )
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
    static void getValue(const DataType &data, Index index, T& value)
    {
        if (index != 0) return;
        value = static_cast<T>(data);
    }

    template<typename T>
    static void setValue(DataType &data, Index index, const T& value )
    {
        if (index != 0) return;
        data = static_cast<DataType>(value);
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

    static const void* getValuePtr(const DataType& data)
    {
        return &data;
    }

    static void* getValuePtr(DataType& data)
    {
        return &data;
    }
};

template<class TDataType>
struct TextTypeInfo
{
    typedef TDataType DataType;
    typedef DataType BaseType;
    typedef DataType ValueType;
    typedef long long ConvType;
    typedef ScalarTypeInfo<TDataType> BaseTypeInfo;
    typedef ScalarTypeInfo<TDataType> ValueTypeInfo;

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
    static sofa::Size size() { return 1; }
    static sofa::Size byteSize() { return 1; }

    static sofa::Size size(const DataType& /*data*/) { return 1; }

    static bool setSize(DataType& /*data*/, sofa::Size /*size*/) { return false; }

    template <typename T>
    static void getValue(const DataType &data, Index index, T& value)
    {
        if (index != 0) return;
        std::istringstream i(data); i >> value;
    }

    template<typename T>
    static void setValue(DataType &data, Index index, const T& value )
    {
        if (index != 0) return;
        std::ostringstream o; o << value; data = o.str();
    }

    static void getValueString(const DataType &data, Index index, std::string& value)
    {
        if (index != 0) return;
        value = data;
    }

    static void setValueString(DataType &data, Index index, const std::string& value )
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

    enum { Size            = static_size * BaseTypeInfo::Size };
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
    static void getValue(const DataType &data, Index index, T& value)
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
                auto n = BaseTypeInfo::size(data[(sofa::Size)i]);
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
    static void setValue(DataType &data, Index index, const T& value )
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
                auto n = BaseTypeInfo::size(data[(sofa::Size)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::setValue(data[(sofa::Size)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void getValueString(const DataType &data, Index index, std::string& value)
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
                auto n = BaseTypeInfo::size(data[(sofa::Size)i]);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(data[(sofa::Size)i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &data, Index index, const std::string& value )
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
                auto n = BaseTypeInfo::size(data[(sofa::Size)i]);
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

template<class TDataType>
struct VectorTypeInfo
{
    typedef TDataType DataType;
    //typedef typename DataType::sofa::Size sofa::Size;
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
    static sofa::Size size()
    {
        return BaseTypeInfo::size();
    }

    static sofa::Size byteSize()
    {
        return ValueTypeInfo::byteSize();
    }

    static sofa::Size size(const DataType& data)
    {
        if (BaseTypeInfo::FixedSize)
            return sofa::Size(data.size()*BaseTypeInfo::size());
        else
        {
            auto n = data.size();
            sofa::Size s = 0;
            for (sofa::Size i=0; i<n; ++i)
                s+= BaseTypeInfo::size(data[(sofa::Size)i]);
            return s;
        }
    }

    static bool setSize(DataType& data, sofa::Size size)
    {
        if (BaseTypeInfo::FixedSize)
        {
            data.resize(size/BaseTypeInfo::size());
            return true;
        }
        return false;
    }

    template <typename T>
    static void getValue(const DataType &data, Index index, T& value)
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
            for (sofa::Size i=0; i<data.size(); ++i)
            {
                auto n = sofa::Size(BaseTypeInfo::size(data[i]));
                if (index < s+n)
                {
                    BaseTypeInfo::getValue(data[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    template<typename T>
    static void setValue(DataType &data, Index index, const T& value )
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
            for (sofa::Size i=0; i<data.size(); ++i)
            {
                auto n = sofa::Size(BaseTypeInfo::size(data[i]));
                if (index < s+n)
                {
                    BaseTypeInfo::setValue(data[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void getValueString(const DataType &data, Index index, std::string& value)
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
            for (sofa::Size i=0; i< sofa::Size(data.size()); ++i)
            {
                auto n = sofa::Size(BaseTypeInfo::size(data[(sofa::Size)i]));
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(data[i], index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &data, Index index, const std::string& value )
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
            for (sofa::Size i=0; i< sofa::Size(data.size()); ++i)
            {
                auto n = sofa::Size(BaseTypeInfo::size(data[i]));
                if (index < s+n)
                {
                    BaseTypeInfo::setValueString(data[i], index-s, value);
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
    static sofa::Size size()
    {
        return BaseTypeInfo::size();
    }

    static sofa::Size byteSize()
    {
        return ValueTypeInfo::byteSize();
    }

    static sofa::Size size(const DataType& data)
    {
        if (BaseTypeInfo::FixedSize)
            return sofa::Size(data.size()*BaseTypeInfo::size());
        else
        {
            sofa::Size s = 0;
            for (typename DataType::const_iterator it = data.begin(), end=data.end(); it!=end; ++it)
                s+= BaseTypeInfo::size(*it);
            return s;
        }
    }

    static bool setSize(DataType& data, sofa::Size /*size*/)
    {
        data.clear(); // we can't "resize" a set, so the only meaningfull operation is to clear it, as values will be added dynamically in setValue
        return true;
    }

    template <typename T>
    static void getValue(const DataType &data, Index index, T& value)
    {
        if (BaseTypeInfo::FixedSize)
        {
            typename DataType::const_iterator it = data.begin();
            for (sofa::Size i=0; i<index/BaseTypeInfo::size(); ++i) ++it;
            BaseTypeInfo::getValue(*it, index%BaseTypeInfo::size(), value);
        }
        else
        {
            sofa::Size s = 0;
            for (typename DataType::const_iterator it = data.begin(), end=data.end(); it!=end; ++it)
            {
                auto n = BaseTypeInfo::size(*it);
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
    static void setValue(DataType &data, Index /*index*/, const T& value )
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

    static void getValueString(const DataType &data, Index index, std::string& value)
    {
        if (BaseTypeInfo::FixedSize)
        {
            typename DataType::const_iterator it = data.begin();
            for (sofa::Size i=0; i<index/BaseTypeInfo::size(); ++i) ++it;
            BaseTypeInfo::getValueString(*it, index%BaseTypeInfo::size(), value);
        }
        else
        {
            sofa::Size s = 0;
            for (typename DataType::const_iterator it = data.begin(), end=data.end(); it!=end; ++it)
            {
                auto n = BaseTypeInfo::size(*it);
                if (index < s+n)
                {
                    BaseTypeInfo::getValueString(*it, index-s, value);
                    break;
                }
                s += n;
            }
        }
    }

    static void setValueString(DataType &data, Index /*index*/, const std::string& value )
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

template<>
struct DataTypeInfo<bool> : public BoolTypeInfo
{
    static const char* name() { return "bool"; }
};

template<>
struct DataTypeInfo<char> : public IntegerTypeInfo<char>
{
    static const char* name() { return "char"; }
};

template<>
struct DataTypeInfo<unsigned char> : public IntegerTypeInfo<unsigned char>
{
    static const char* name() { return "unsigned char"; }
};

template<>
struct DataTypeInfo<short> : public IntegerTypeInfo<short>
{
    static const char* name() { return "short"; }
};

template<>
struct DataTypeInfo<unsigned short> : public IntegerTypeInfo<unsigned short>
{
    static const char* name() { return "unsigned short"; }
};

template<>
struct DataTypeInfo<int> : public IntegerTypeInfo<int>
{
    static const char* name() { return "int"; }
};

template<>
struct DataTypeInfo<unsigned int> : public IntegerTypeInfo<unsigned int>
{
    static const char* name() { return "unsigned int"; }
};

template<>
struct DataTypeInfo<long> : public IntegerTypeInfo<long>
{
    static const char* name() { return "long"; }
};

template<>
struct DataTypeInfo<unsigned long> : public IntegerTypeInfo<unsigned long>
{
    static const char* name() { return "unsigned long"; }
};

template<>
struct DataTypeInfo<long long> : public IntegerTypeInfo<long long>
{
    static const char* name() { return "long long"; }
};

template<>
struct DataTypeInfo<unsigned long long> : public IntegerTypeInfo<unsigned long long>
{
    static const char* name() { return "unsigned long long"; }
};

template<>
struct DataTypeInfo<float> : public ScalarTypeInfo<float>
{
    static const char* name() { return "float"; }
};

template<>
struct DataTypeInfo<double> : public ScalarTypeInfo<double>
{
    static const char* name() { return "double"; }
};

template<>
struct DataTypeInfo<std::string> : public TextTypeInfo<std::string>
{
    static const char* name() { return "string"; }

    static const void* getValuePtr(const std::string& data) { return &data[0]; }
    static void* getValuePtr(std::string& data) { return &data[0]; }
};

template<class T, sofa::Size N>
struct DataTypeInfo< sofa::helper::fixed_array<T,N> > : public FixedArrayTypeInfo<sofa::helper::fixed_array<T,N> >
{
    static std::string name() { std::ostringstream o; o << "fixed_array<" << DataTypeName<T>::name() << "," << N << ">"; return o.str(); }
};

template<class T, class Alloc>
struct DataTypeInfo< std::vector<T,Alloc> > : public VectorTypeInfo<std::vector<T,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "std::vector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<class T, class Alloc>
struct DataTypeInfo< sofa::helper::vector<T,Alloc> > : public VectorTypeInfo<sofa::helper::vector<T,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "vector<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

// vector<bool> is a bitset, cannot get a pointer to the values
template<class Alloc>
struct DataTypeInfo< sofa::helper::vector<bool,Alloc> > : public VectorTypeInfo<sofa::helper::vector<bool,Alloc> >
{
    enum { SimpleLayout = 0 };

    static std::string name() { std::ostringstream o; o << "vector<bool>"; return o.str(); }

    static const void* getValuePtr(const sofa::helper::vector<bool,Alloc>& /*data*/) { return nullptr; }
    static void* getValuePtr(sofa::helper::vector<bool,Alloc>& /*data*/) { return nullptr; }
};

// Cannot use default impl of VectorTypeInfo for non-fixed size BaseTypes
template<class Alloc>
struct DataTypeInfo< sofa::helper::vector<std::string,Alloc> > : public VectorTypeInfo<sofa::helper::vector<std::string,Alloc> >
{
    static std::string name() { return "vector<string>"; }

    // BaseType size is not fixed. Returning 1
    static sofa::Size size() { return 1; }

    // Total number of elements in the vector
    static sofa::Size size(const sofa::helper::vector<std::string,Alloc>& data) { return sofa::Size(data.size()); }

    // Resizes the vector
    static bool setSize(sofa::helper::vector<std::string,Alloc>& data, sofa::Size size) { data.resize(size); return true; }

    // Sets the value for element at index `index`
    static void setValueString(sofa::helper::vector<std::string,Alloc>& data, Index index, const std::string& value)
    {
        if (data.size() <= index)
            data.resize(index + 1);
        data[index] = value;
    }

    // Gets the value for element at index `index`
    static void getValueString(const sofa::helper::vector<std::string,Alloc>& data, Index index, std::string& value)
    {
        if (data.size() <= index)
            msg_error("DataTypeInfo<helper::vector<std::string>") << "Index out of bounds for getValueString";
        else
            value = data[index];
    }
};

template<class T, class Compare, class Alloc>
struct DataTypeInfo< std::set<T,Compare,Alloc> > : public SetTypeInfo<std::set<T,Compare,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "std::set<" << DataTypeName<T>::name() << ">"; return o.str(); }
};

template<>
struct DataTypeInfo< sofa::helper::types::RGBAColor > : public FixedArrayTypeInfo<sofa::helper::fixed_array<float,4>>
{
    static std::string name() { return "RGBAColor"; }
};

} // namespace defaulttype

} // namespace sofa

#endif  // SOFA_DEFAULTTYPE_DATATYPEINFO_H
