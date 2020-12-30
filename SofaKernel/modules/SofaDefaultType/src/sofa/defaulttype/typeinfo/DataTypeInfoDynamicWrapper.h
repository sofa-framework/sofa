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
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfo.h>
#include <sofa/defaulttype/TypeInfoID.h>
#include <string>

namespace sofa::defaulttype
{

/**
 * @brief add a compatibility layer to supper the new existing GetTypeName.
 */
template<class T>
class HasGetTypeName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::GetTypeName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};




/**
    @brief wrap a DataTypeInfo type_traits in an AbstractTypeInfo

    Example of use:
    \code{.cpp}

    /// This is a custom type
    class MyType {};

    /// This is a static DataTypeInfo that you can expose to in traits oriented code.
    template<>
    class DataTypeInfo<MyType> { }

    /// Wrap the DataTypeInfo type in a dynamic version of type info (thus inheriting from AbstractTypeInfo).
    AbstractTypeInfo* nfo = DataTypeInfoDynamicWrapper<DataTypeInfo<MyType>>();


    /// Wrap the DataTypeInfo type in a dynamic version of type info (thus inheriting from AbstractTypeInfo).
    TypeInfoRegistry::Set(TypeInfoId::getNewId<MyType>(), info);
    \endcode
**/
template<class Info>
class DataTypeInfoDynamicWrapper : public AbstractTypeInfo, public Info
{
public:
    typedef typename Info::DataType DataType;

    const AbstractTypeInfo* BaseType() const override
    {
        return DataTypeInfoDynamicWrapper<DataTypeInfo<typename Info::BaseType>>::get();
    }
    const AbstractTypeInfo* ValueType() const override
    {
        return DataTypeInfoDynamicWrapper<DataTypeInfo<typename Info::ValueType>>::get();
    }

    static AbstractTypeInfo* get() { static DataTypeInfoDynamicWrapper<Info> t; return &t; }

    const TypeInfoId& getBaseTypeId() const override { return TypeInfoId::GetTypeId<typename Info::BaseType>(); }
    const TypeInfoId& getValueTypeId() const override { return TypeInfoId::GetTypeId<typename Info::ValueType>(); }

    std::string name() const override { return Info::name(); }
    std::string getTypeName() const override {return Info::name();}

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
    DataTypeInfoDynamicWrapper() {}
};

} /// namespace sofa::defaulttype
