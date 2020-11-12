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
#include <sofa/defaulttype/config.h>
#include <sofa/helper/fixed_array.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/set.h>
#include <sstream>
#include <typeinfo>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
#include <sofa/helper/NameDecoder.h>

namespace sofa::helper
{
template <class T, class MemoryManager >
class vector;
}

namespace sofa::defaulttype
{

class WillBeDeprecated
{
public:

    template<class TT>
    [[deprecated("This will be deprecated soon.... please update your code...XXXX")]]
    static std::string getDeprecatedName(const std::string& n){ return n; }
};

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

    enum { ValidInfo       = 0 /**< 1 if this type has valid infos*/ };

//    static std::string GetName(){ return WillBeDeprecated::getDeprecatedName<TDataType>("undefined"); }

//    enum [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { FixedSize       = 0 /**< 1 if this type has a fixed size*/ };
//    enum [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { ZeroConstructor =  0 /**< 1 if the constructor is equivalent to setting memory to 0*/ };
//    enum [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { SimpleCopy      = 0 /**< 1 if copying the data can be done with a memcpy*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { SimpleLayout    = 0 /**< 1 if the layout in memory is simply N values of the same base type*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { Integer         = 0 /**< 1 if this type uses integer values*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { Scalar          = 0 /**< 1 if this type uses scalar values*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { Text            = 0 /**< 1 if this type uses text values*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { CopyOnWrite     = 0 /**< 1 if this type uses copy-on-write. The memory is shared with its source Data while only the source is changing (and the source modifications are then visible in the current Data). As soon as modifications are applied to the current Data, it will allocate its own value, and no longer shares memory with the source.*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { Container       = 0 /**< 1 if this type is a container*/ };
//    enum  [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    { Size = 0 /**< largest known fixed size for this type, as returned by size() */ };

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static size_t size() { return 1; }

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static size_t byteSize() { return 1; }

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static size_t size(const DataType& /*data*/) { return 1; }

//    template <typename T>
//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static void getValue(const DataType& /*data*/, size_t /*index*/, T& /*value*/)
//    {}

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static bool setSize(DataType& /*data*/, size_t /*size*/) { return false; }

//    template<typename T>
//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static void setValue(DataType& /*data*/, size_t /*index*/, const T& /*value*/){}

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static void getValueString(const DataType& /*data*/, size_t /*index*/, std::string& /*value*/){}

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static const void* getValuePtr(const TDataType& data)
//    {
//        SOFA_UNUSED(data);
//        return nullptr;
//    }

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static void* getValuePtr(TDataType& data)
//    {
//        SOFA_UNUSED(data);
//        return nullptr;
//    }

//    [[deprecated("Using values of a not fully defined component is not allowed since PR#xxxx")]]
//    static void setValueString(DataType &data, size_t /*index*/, const std::string& value)
//    {
//        SOFA_UNUSED(data);
//        SOFA_UNUSED(index);
//        SOFA_UNUSED(value);
//    }


};


template<class TDataType>
struct PartialDataTypeInfo
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

    static std::string GetName(){ return WillBeDeprecated::getDeprecatedName<TDataType>("undefined"); }

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

    static size_t size() { return 0; }
    static size_t byteSize() { return 0; }
    static size_t size(const DataType& /*data*/) { return 1; }

    template <typename T>
    static void getValue(const DataType& /*data*/, size_t /*index*/, T& /*value*/)
    {}

    static bool setSize(DataType& /*data*/, size_t /*size*/) { return false; }

    template<typename T>
    static void setValue(DataType& /*data*/, size_t /*index*/, const T& /*value*/){}

    static void getValueString(const DataType& /*data*/, size_t /*index*/, std::string& /*value*/){}

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

    static void setValueString(DataType &data, size_t /*index*/, const std::string& value)
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        SOFA_UNUSED(value);
    }


};

//TODO(dmarchal: 01/04/2020) This is for compatibility layer, remove it after 01/07/2021
template<class T>
class HasNewGetName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::GetName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

template<class T>
class HasNewGetTypeName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::GetTypeName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};



/// Abstract type traits class
template<typename Info>
class VirtualTypeInfoA : public AbstractTypeInfo
{
public:
    typedef typename Info::DataType DataType;

    static VirtualTypeInfoA* get() { static VirtualTypeInfoA<Info> t; return &t; }
    const AbstractTypeInfo* getBaseType() const override  { return VirtualTypeInfoA<DataTypeInfo<typename Info::BaseType>>::get(); }
    const AbstractTypeInfo* getValueType() const override { return VirtualTypeInfoA<DataTypeInfo<typename Info::ValueType>>::get(); }

    virtual std::string getName() const override
    {
        if constexpr(HasNewGetName<Info>::value)
                return Info::GetName();
        else
        {
            return WillBeDeprecated::getDeprecatedName<Info>(Info::name());
        }
    }
    virtual std::string getTypeName() const override
    {
        if constexpr(HasNewGetTypeName<Info>::value)
                return Info::GetTypeName();
        else
        {
            return WillBeDeprecated::getDeprecatedName<Info>(Info::name());
        }
    }

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

/// Abstract type traits class
template<class T>
class IncompleteTypeInfo : public AbstractTypeInfo
{
public:
    /// Template parameter.
    typedef T DataType;

    /// If the type is a container, this the type of the values inside this
    /// container, otherwise this is DataType.
    typedef DataType BaseType;

    /// Type of the final atomic values (i.e. the values indexed by getValue()).
    typedef DataType ValueType;

    /// TypeInfo for BaseType
    typedef DataTypeInfo<BaseType> BaseTypeInfo;

    /// TypeInfo for ValueType
    typedef DataTypeInfo<ValueType> ValueTypeInfo;

    const AbstractTypeInfo* getBaseType() const override  { return nullptr; }
    const AbstractTypeInfo* getValueType() const override { return nullptr; }

    virtual std::string getTypeName() const override { return sofa::helper::NameDecoder::decodeFullName(typeid(T)); }
    virtual std::string getName() const override { return sofa::helper::NameDecoder::decodeFullName(typeid(T)); }

    bool ValidInfo() const override       { return 0; }
    bool FixedSize() const override       { return 0; }
    bool ZeroConstructor() const override { return 0; }
    bool SimpleCopy() const override      { return 0; }
    bool SimpleLayout() const override    { return 0; }
    bool Integer() const override         { return 0; }
    bool Scalar() const override          { return 0; }
    bool Text() const override            { return 0; }
    bool CopyOnWrite() const override     { return 0; }
    bool Container() const override       { return 0; }

    size_t size() const override
    {
        return 0;
    }
    size_t byteSize() const override
    {
        return 0;
    }
    size_t size(const void* data) const override
    {
        SOFA_UNUSED(data);
        return 0;
    }
    bool setSize(void* data, size_t size) const override
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(size);
        return 0;
    }

    long long getIntegerValue(const void* data, size_t index) const override
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        return 0;
    }

    double    getScalarValue (const void* data, size_t index) const override
    {

        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        return 0;
    }

    virtual std::string getTextValue   (const void* data, size_t index) const override
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        return "";
    }

    void setIntegerValue(void* data, size_t index, long long value) const override
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        SOFA_UNUSED(value);
    }

    void setScalarValue (void* data, size_t index, double value) const override
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        SOFA_UNUSED(value);
    }

    virtual void setTextValue(void* data, size_t index, const std::string& value) const override
    {
        SOFA_UNUSED(data);
        SOFA_UNUSED(index);
        SOFA_UNUSED(value);
    }
    const void* getValuePtr(const void* data) const override
    {
        SOFA_UNUSED(data);
        return nullptr;
    }
    void* getValuePtr(void* data) const override
    {
        SOFA_UNUSED(data);
        return nullptr;
    }

    virtual const std::type_info* type_info() const override { return &typeid(T); }
};

class BaseDataTypeId
{
public:
    int id;
    const std::type_info& nfo;
    BaseDataTypeId(int id_, const std::type_info& nfo_):
        id(id_),
        nfo(nfo_){}

    static int getNewId()
    {
        static int value = 0;
        return value++;
    }
};

template<class T>
class AbstractTypeInfoCreator
{
public:

    static AbstractTypeInfo* get()
    {
        if constexpr( DataTypeInfo<T>::ValidInfo )
        {
            return VirtualTypeInfoA<DataTypeInfo<T>>::get();
        }
        else
        {
            static IncompleteTypeInfo<T> t;
            return &t;
        }
    }
};

template<class T>
class DataTypeId : public BaseDataTypeId
{
public:


    DataTypeId(int id_, const std::type_info& nfo_) : BaseDataTypeId(id_, nfo_){}


    static const DataTypeId& getTypeId()
    {
        static DataTypeId typeId(BaseDataTypeId::getNewId(), typeid(T));
        return typeId;
    }



//    static const sofa::defaulttype::AbstractTypeInfo* GetDataTypeInfo()
//    {
//        static int a = DataTypeId<T>::m_register;
//        SOFA_UNUSED(a);
//        static const sofa::defaulttype::AbstractTypeInfo* typeinfo {nullptr};
//        if(typeinfo==nullptr)
//        {
//            /// We don't cache valid info;
//            auto tmpinfo = sofa::defaulttype::DataTypeInfoRegistry::Get(sofa::defaulttype::DataTypeId<T>::getTypeId());
//            if(!tmpinfo->ValidInfo())
//                return tmpinfo;
//            typeinfo = tmpinfo;
//        }
//        return typeinfo;
//    }
};

template<class T>
class BaseDataTypeInfo
{
public:
    static int m_id;

};

template<class T>
int BaseDataTypeInfo<T>::m_id;


template<class TDataType>
struct DataTypeInfo : public BaseDataTypeInfo<TDataType>, public DefaultDataTypeInfo<TDataType>
{
public:
    static AbstractTypeInfo* get(){}
};

class TypeInfo
{
public:
    template<class T>
    static constexpr int Initialize(){
        doRegister<T>();
        return sofa::defaulttype::DataTypeId<T>::getTypeId().id;
    }

    template<class T>
    static AbstractTypeInfo* doRegister()
    {
        static IncompleteTypeInfo<T> info;
        sofa::defaulttype::DataTypeInfoRegistry::Set(sofa::defaulttype::DataTypeId<T>::getTypeId(),
                                                     &info, sofa_tostring(SOFA_TARGET));
        return &info;
    }

    template<class T>
    static const sofa::defaulttype::AbstractTypeInfo* Get()
    {
        static const sofa::defaulttype::AbstractTypeInfo* typeinfo {doRegister<T>()};
        if(!typeinfo->ValidInfo())
        {
            /// We don't cache valid info;
            auto tmpinfo = sofa::defaulttype::DataTypeInfoRegistry::Get(sofa::defaulttype::DataTypeId<T>::getTypeId());
            if(!tmpinfo->ValidInfo())
                return tmpinfo;
            typeinfo = tmpinfo;
        }

        return typeinfo;
    }
};

/// Type name template: default to using DataTypeInfo::name(), but can be overriden for types with shorter typedefs
template<class TDataType>
struct DataTypeName : public DataTypeInfo<TDataType>
{
public:
    static std::string name(){ return DataTypeInfo<TDataType>::GetName();  }
};

#define REGISTER_MSG_PASTER(x,y) x ## _ ## y
#define REGISTER_UNIQUE_NAME_GENERATOR(x,y)  REGISTER_MSG_PASTER(x,y)
#define REGISTER_TYPE_INFO_CREATOR(theTypeName) static int REGISTER_UNIQUE_NAME_GENERATOR(_theTypeName_ , __LINE__) = DataTypeInfoRegistry::Set(DataTypeId<theTypeName>::getTypeId(), \
    VirtualTypeInfoA< DataTypeInfo<theTypeName>>::get(),\
    sofa_tostring(SOFA_TARGET));
#define REGISTER_TYPE_INFO_CREATOR2(theTypeName, ext) template<> AbstractTypeInfo* AbstractTypeInfoCreator< theTypeName ext >::get() {return VirtualTypeInfoA< DataTypeInfo<theTypeName ext> >::get();}

}/// namespace sofa::defaulttype
