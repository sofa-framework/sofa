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

#include <vector>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/set.h>
#include <sstream>
#include <typeinfo>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/defaulttype/AbstractTypeInfo.h>

namespace sofa::helper
{
template <class T, class MemoryManager >
class vector;
}

namespace sofa::defaulttype
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
    //enum { FixedSize       = 0 /**< 1 if this type has a fixed size*/ };
    //enum { ZeroConstructor = 0 /**< 1 if the constructor is equivalent to setting memory to 0*/ };
    //enum { SimpleCopy      = 0 /**< 1 if copying the data can be done with a memcpy*/ };
    //enum { SimpleLayout    = 0 /**< 1 if the layout in memory is simply N values of the same base type*/ };
    //enum { Integer         = 0 /**< 1 if this type uses integer values*/ };
    //enum { Scalar          = 0 /**< 1 if this type uses scalar values*/ };
    //enum { Text            = 0 /**< 1 if this type uses text values*/ };
    //enum { CopyOnWrite     = 0 /**< 1 if this type uses copy-on-write. The memory is shared with its source Data while only the source is changing (and the source modifications are then visible in the current Data). As soon as modifications are applied to the current Data, it will allocate its own value, and no longer shares memory with the source.*/ };
    //enum { Container       = 0 /**< 1 if this type is a container*/ };
    //enum { Size = 1 /**< largest known fixed size for this type, as returned by size() */ };

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
        return nullptr;
    }

    static void* getValuePtr(DataType& /*type*/)
    {
        return nullptr;
    }

    //static const char* name() { return "unknown"; }

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

    virtual std::string name() const override { return Info::name(); }

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

    size_t size() const override
    {
        return Info::size();
    }
    size_t byteSize() const override
    {
        return Info::byteSize();
    }
    size_t size(const void* data) const override
    {
        return Info::size(*(const DataType*)data);
    }
    bool setSize(void* data, size_t size) const override
    {
        return Info::setSize(*(DataType*)data, size);
    }

    long long getIntegerValue(const void* data, size_t index) const override
    {
        long long v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    double    getScalarValue (const void* data, size_t index) const override
    {
        double v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    virtual std::string getTextValue   (const void* data, size_t index) const override
    {
        std::string v;
        Info::getValueString(*(const DataType*)data, index, v);
        return v;
    }

    void setIntegerValue(void* data, size_t index, long long value) const override
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    void setScalarValue (void* data, size_t index, double value) const override
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    virtual void setTextValue(void* data, size_t index, const std::string& value) const override
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


/// Abstract type traits class
template<typename Info>
class VirtualTypeInfoA : public AbstractTypeInfo
{
public:
    typedef typename Info::DataType DataType;
    static VirtualTypeInfoA<Info>* get() { static VirtualTypeInfoA<Info> t; return &t; }
    
    //typedef typename Info::BaseType BaseType;
    //typedef typename Info::ValueType ValueType;

    const AbstractTypeInfo* BaseType() const override  { return VirtualTypeInfo<typename Info::BaseType>::get(); }
    const AbstractTypeInfo* ValueType() const override { return VirtualTypeInfo<typename Info::ValueType>::get(); }

    virtual std::string name() const override { return Info::name(); }

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

    size_t size() const override
    {
        return Info::size();
    }
    size_t byteSize() const override
    {
        return Info::byteSize();
    }
    size_t size(const void* data) const override
    {
        return Info::size(*(const DataType*)data);
    }
    bool setSize(void* data, size_t size) const override
    {
        return Info::setSize(*(DataType*)data, size);
    }

    long long getIntegerValue(const void* data, size_t index) const override
    {
        long long v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    double    getScalarValue (const void* data, size_t index) const override
    {
        double v = 0;
        Info::getValue(*(const DataType*)data, index, v);
        return v;
    }

    virtual std::string getTextValue   (const void* data, size_t index) const override
    {
        std::string v;
        Info::getValueString(*(const DataType*)data, index, v);
        return v;
    }

    void setIntegerValue(void* data, size_t index, long long value) const override
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    void setScalarValue (void* data, size_t index, double value) const override
    {
        Info::setValue(*(DataType*)data, index, value);
    }

    virtual void setTextValue(void* data, size_t index, const std::string& value) const override
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
    VirtualTypeInfoA() {}
};


template<class TDataType>
struct DataTypeInfo : DefaultDataTypeInfo<TDataType> { };



/// Type name template: default to using DataTypeInfo::name(), but can be overriden for types with shorter typedefs
template<class TDataType>
struct DataTypeName : public DataTypeInfo<TDataType>
{
};

}/// namespace sofa::defaulttype
