/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sstream>
#include <typeinfo>
#include <sofa/helper/logging/Messaging.h>

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
    enum { Size = 1 /**< largest known fixed size for this type, as returned by size() */ };

    // \}

    static size_t size() { return 1; }
    static size_t byteSize() { return 1; }

    static size_t size(const DataType& /*data*/) { return 1; }

    template <typename T>
    static void getValue(const DataType& /*data*/, size_t /*index*/, T& /*value*/)
    {
    }

    static bool setSize(DataType& /*data*/, size_t /*size*/) { return false; }

    template<typename T>
    static void setValue(DataType& /*data*/, size_t /*index*/, const T& /*value*/)
    {
    }

    static void getValueString(const DataType& /*data*/, size_t /*index*/, std::string& /*value*/)
    {
    }

    static void setValueString(DataType& /*data*/, size_t /*index*/, const std::string& /*value*/)
    {
    }

    // mtournier: wtf is this supposed to do?
    // mtournier: wtf is this not returning &type?
    static const void* getValuePtr(const DataType& /*type*/)
    {
        return NULL;
    }

    static void* getValuePtr(DataType& /*type*/)
    {
        return NULL;
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


/** Information about the type of a value stored in a Data.

    %AbstractTypeInfo is part of the introspection/reflection capabilities of
    the Sofa scene graph API. It provides information about the type of the
    content of Data objects (Is it a simple type?  A container? How much memory
    should be allocated to copy it?), and allows manipulating Data generically,
    without knowing their exact type.

    This class is primarily used to copy information accross BaseData objects,
    for example when there exists a link between two instances of BaseData.
    E.g. this mecanism allows you to copy the content of a Data<vector<int>>
    into a Data<vector<double>>, because there is an acceptable conversion
    between integer and double, and because both Data use a resizable container.

    <h4>Using TypeInfo</h4>

    Use BaseData::getValueTypeInfo() to get a pointer to an AbtractTypeInfo, and
    BaseData::getValueVoidPtr() to get a pointer to the content of a Data. You
    can then use the methods of AbtractTypeInfo to access the Data generically.

    Very basic example:
    \code{.cpp}
    BaseData *data = getADataFromSomewhere();
    const AbstractTypeInfo *typeinfo = data->getValueTypeInfo();
    const void* ptr = data->getValueVoidPtr();
    for (int i = 0 ; i < typeinfo->size(ptr) ; i++)
    std::string value = typeinfo->getTextValue(ptr, 0);
    \endcode

    <h4>Note about size and indices</h4>

    All the getValue() and setValue() methods take an index as a parameter,
    which means that every type is abstracted to a one-dimensional container.
    See the detailed description of DataTypeInfo for more explanations.

    \see DataTypeInfo provides similar mechanisms to manipulate Data objects
    generically in template code.
*/
class AbstractTypeInfo
{
public:
    /// If the type is a container, returns the TypeInfo for the type of the
    /// values inside this container.
    /// For example, if the type is `fixed_array<fixed_array<int, 2> 3>`, it
    /// returns the TypeInfo for `fixed_array<int, 2>`.
    virtual const AbstractTypeInfo* BaseType() const = 0;
    /// Returns the TypeInfo for the type of the values accessible by the
    /// get*Value() functions.
    /// For example, if the type is `fixed_array<fixed_array<int, 2> 3>`, it
    /// returns the TypeInfo for `int`.
    virtual const AbstractTypeInfo* ValueType() const = 0;

    /// \brief Returns the name of this type.
    virtual std::string name() const = 0;

    /// True iff the TypeInfo for this type contains valid information.
    virtual bool ValidInfo() const = 0;
    /// True iff this type has a fixed size.
    virtual bool FixedSize() const = 0;
    /// True iff the default constructor of this type is equivalent to setting the memory to 0.
    virtual bool ZeroConstructor() const = 0;
    /// True iff copying the data can be done with a memcpy().
    virtual bool SimpleCopy() const = 0;
    /// True iff the layout in memory is simply N values of the same base type.
    virtual bool SimpleLayout() const = 0;
    /// True iff this type uses integer values.
    virtual bool Integer() const = 0;
    /// True iff this type uses scalar values.
    virtual bool Scalar() const = 0;
    /// True iff this type uses text values.
    virtual bool Text() const = 0;
    /// True iff this type uses copy-on-write.
    virtual bool CopyOnWrite() const = 0;
    /// True iff this type is a container of some sort.
    ///
    /// That is, if it can contain several values. In particular, strings are
    /// not considered containers.
    virtual bool Container() const = 0;

    /// The size of this type, in number of elements.
    /// For example, the size of a `fixed_array<fixed_array<int, 2> 3>` is 6,
    /// and those six elements are conceptually numbered from 0 to 5.  This is
    /// relevant only if FixedSize() is true.
    virtual size_t size() const = 0;
    /// The size in bytes of the ValueType
    virtual size_t byteSize() const = 0;

    /// The size of \a data, in number of elements.
    virtual size_t size(const void* data) const = 0;
    /// Resize \a data to \a size elements, if relevant.

    /// But resizing is not always relevant, for example:
    /// - nothing happens if FixedSize() is true;
    /// - sets can't be resized; they are cleared instead;
    /// - nothing happens for vectors containing resizable values (i.e. when
    ///   BaseType()::FixedSize() is false), because of the "single index"
    ///   abstraction;
    ///
    /// Returns true iff the data was resizable
    virtual bool setSize(void* data, size_t size) const = 0;

    /// Get the value at \a index of \a data as an integer.
    /// Relevant only if this type can be casted to `long long`.
    virtual long long   getIntegerValue(const void* data, size_t index) const = 0;
    /// Get the value at \a index of \a data as a scalar.
    /// Relevant only if this type can be casted to `double`.
    virtual double      getScalarValue (const void* data, size_t index) const = 0;
    /// Get the value at \a index of \a data as a string.
    virtual std::string getTextValue   (const void* data, size_t index) const = 0;

    /// Set the value at \a index of \a data from an integer value.
    virtual void setIntegerValue(void* data, size_t index, long long value) const = 0;
    /// Set the value at \a index of \a data from a scalar value.
    virtual void setScalarValue (void* data, size_t index, double value) const = 0;
    /// Set the value at \a index of \a data from a string value.
    virtual void setTextValue(void* data, size_t index, const std::string& value) const = 0;

    /// Get a read pointer to the underlying memory
    /// Relevant only if this type is SimpleLayout
    virtual const void* getValuePtr(const void* type) const = 0;

    /// Get a write pointer to the underlying memory
    /// Relevant only if this type is SimpleLayout
    virtual void* getValuePtr(void* type) const = 0;

    /// Get the type_info for this type.
    virtual const std::type_info* type_info() const = 0;

protected: // only derived types can instantiate this class
    AbstractTypeInfo() {}
    virtual ~AbstractTypeInfo() {}

private: // copy constructor or operator forbidden
    AbstractTypeInfo(const AbstractTypeInfo&) {}
    void operator=(const AbstractTypeInfo&) {}
};

/// Abstract type traits class
template<class TDataType>
class VirtualTypeInfo : public AbstractTypeInfo
{
public:
    typedef TDataType DataType;
    typedef DataTypeInfo<DataType> Info;

    static VirtualTypeInfo* get() { static VirtualTypeInfo<DataType> t; return &t; }

    virtual const AbstractTypeInfo* BaseType() const  { return VirtualTypeInfo<typename Info::BaseType>::get(); }
    virtual const AbstractTypeInfo* ValueType() const { return VirtualTypeInfo<typename Info::ValueType>::get(); }

    virtual std::string name() const { return DataTypeName<DataType>::name(); }

    virtual bool ValidInfo() const       { return Info::ValidInfo; }
    virtual bool FixedSize() const       { return Info::FixedSize; }
    virtual bool ZeroConstructor() const { return Info::ZeroConstructor; }
    virtual bool SimpleCopy() const      { return Info::SimpleCopy; }
    virtual bool SimpleLayout() const    { return Info::SimpleLayout; }
    virtual bool Integer() const         { return Info::Integer; }
    virtual bool Scalar() const          { return Info::Scalar; }
    virtual bool Text() const            { return Info::Text; }
    virtual bool CopyOnWrite() const     { return Info::CopyOnWrite; }
    virtual bool Container() const       { return Info::Container; }

    virtual size_t size() const
    {
        return Info::size();
    }
    size_t byteSize() const
    {
        return Info::byteSize();
    }
    virtual size_t size(const void* data) const
    {
        return Info::size(*(const DataType*)data);
    }
    virtual bool setSize(void* data, size_t size) const
    {
        return Info::setSize(*(DataType*)data, size);
    }

    virtual long long getIntegerValue(const void* data, size_t index) const
    {
        long long v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    virtual double    getScalarValue (const void* data, size_t index) const
    {
        double v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    virtual std::string getTextValue   (const void* data, size_t index) const
    {
        std::string v;
        Info::getValueString(*(const DataType*)data, index, v);
        return v;
    }

    virtual void setIntegerValue(void* data, size_t index, long long value) const
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    virtual void setScalarValue (void* data, size_t index, double value) const
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    virtual void setTextValue(void* data, size_t index, const std::string& value) const
    {
        Info::setValueString(*(DataType*)data, index, value);
    }
    virtual const void* getValuePtr(const void* data) const
    {
        return Info::getValuePtr(*(const DataType*)data);
    }
    virtual void* getValuePtr(void* data) const
    {
        return Info::getValuePtr(*(DataType*)data);
    }

    virtual const std::type_info* type_info() const { return &typeid(DataType); }


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
        data = static_cast<DataType>(value);
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
        data = static_cast<DataType>(value);
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
        return NULL;
    }

    static void* getValuePtr(DataType& /*data*/)
    {
        return NULL;
    }
};

template<class TDataType, int static_size = TDataType::static_size>
struct FixedArrayTypeInfo
{
    typedef TDataType DataType;
    typedef typename DataType::size_type size_type;
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
    static size_t size()
    {
        return DataType::size() * BaseTypeInfo::size();
    }

    static size_t byteSize()
    {
        return ValueTypeInfo::byteSize();
    }

    static size_t size(const DataType& data)
    {
        if (FixedSize)
            return size();
        else
        {
            size_t s = 0;
            for (size_t i=0; i<DataType::size(); ++i)
                s+= BaseTypeInfo::size(data[(size_type)i]);
            return s;
        }
    }

    static bool setSize(DataType& data, size_t size)
    {
        if (!FixedSize)
        {
            size /= DataType::size();
            for (size_t i=0; i<DataType::size(); ++i)
                if( !BaseTypeInfo::setSize(data[(size_type)i], size) ) return false;
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
            for (size_t i=0; i<DataType::size(); ++i)
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
            for (size_t i=0; i<DataType::size(); ++i)
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
            for (size_t i=0; i<DataType::size(); ++i)
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
            for (size_t i=0; i<DataType::size(); ++i)
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
        return NULL;
    }

    static void* getValuePtr(DataType& /*data*/)
    {
    return NULL;
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

template<class T, std::size_t N>
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

    static const void* getValuePtr(const sofa::helper::vector<bool,Alloc>& /*data*/) { return NULL; }
    static void* getValuePtr(sofa::helper::vector<bool,Alloc>& /*data*/) { return NULL; }
};

template<class T, class Compare, class Alloc>
struct DataTypeInfo< std::set<T,Compare,Alloc> > : public SetTypeInfo<std::set<T,Compare,Alloc> >
{
    static std::string name() { std::ostringstream o; o << "std::set<" << DataTypeName<T>::name() << ">"; return o.str(); }
};


} // namespace defaulttype

} // namespace sofa

#endif  // SOFA_DEFAULTTYPE_DATATYPEINFO_H
