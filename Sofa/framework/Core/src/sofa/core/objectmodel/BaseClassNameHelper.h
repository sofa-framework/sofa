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

#include <sofa/core/config.h>
#include <sofa/helper/NameDecoder.h>
#include <sofa/core/fwd.h>

namespace sofa::defaulttype
{
    template<class T> struct DataTypeInfo;
}

namespace sofa::core::objectmodel
{

template<class T>
class HasGetCustomTemplateName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::GetCustomTemplateName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

template<class T>
class HasGetCustomClassName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::GetCustomClassName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

template<class T>
class HasName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::Name) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

template<class T>
class HasDataTypeInfo
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&sofa::defaulttype::DataTypeInfo<C>::name) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

//TODO(dmarchal: 01/04/2020) This is for compatibility layer, remove it after 01/01/2021
template<class T>
class HasGetDefaultTemplateName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::GetDefaultTemplateName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

//TODO(dmarchal: 01/04/2020) This is for compatibility layer, remove it after 01/01/2021
template<class T>
class HasDeprecatedTemplateName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::templateName) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

template<class T>
class HasDeprecatedShortName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test(decltype(&C::template shortName<C>));
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

//TODO(dmarchal: 01/04/2020) This is for compatibility layer, remove it after 01/01/2021
template<class T>
class HasDeprecatedClassName
{
    typedef char YesType[1];
    typedef char NoType[2];

    template<typename C> static YesType& test( decltype (&C::className) );
    template<typename C> static NoType& test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(YesType) };
};

template<typename T>
std::string GetSofaTypeTemplateName(const std::string prefix="");

template<typename T1, typename T2, typename ...Ts>
std::string GetSofaTypeTemplateName(const std::string prefix="");

class BaseClassNameHelper
{
public:
    /// Helper method to get the type name
    template<class T>
    static std::string getTypeName()
    {
        return sofa::helper::NameDecoder::decodeTypeName(typeid(T));
    }

    /// Helper method to get the namespace name
    template<class T>
    static std::string getNamespaceName()
    {
        return sofa::helper::NameDecoder::decodeNamespaceName(typeid(T));
    }

    /// Helper method to get the class name
    template<class T>
    static std::string getClassName()
    {
        return getOverridableClassName<T>();
    }

    /// Helper method to get the template name
    template<class T>
    static std::string getTemplateName()
    {
        return getOverridableTemplateName<T>();
    }

    /// Helper method to get the template name
    template<class T>
    static std::string getShortName()
    {
        return getOverridableShortName<T>();
    }

    template <typename T>
    struct DefaultTypeTemplateName {
        static std::string Get() { return ""; }
    };

    template<template <typename, typename...> class C, typename T1, typename ...Ts>
    struct DefaultTypeTemplateName<C<T1,Ts...>>{
        static std::string Get()
        {
            return GetSofaTypeTemplateName<T1, Ts...>();
        }
    };

private:
    template<class T>
    static const std::string getOverridableShortName()
    {
        /// If there is a Get CustomClassName method in T we use it to return the name
        if constexpr( HasDeprecatedShortName<T>::value )
        {
            T* ptr{nullptr};
            const std::string n = T::template shortName<T>(ptr);
            return n;
        }
        return sofa::helper::NameDecoder::shortName(getOverridableClassName<T>());
    }

    template<class T>
    static const std::string getOverridableClassName()
    {
        /// If there is a Get CustomClassName method in T we use it to return the name
        if constexpr (HasGetCustomClassName<T>::value)
                return T::GetCustomClassName();

        //TODO(dmarchal 01/04/2020): compatibility layer, remove after 01/01/2020
        // use the className method.
        if constexpr (HasDeprecatedClassName<T>::value)
        {
            T* ptr {nullptr};
            const std::string& n = T::className(ptr);
            return n;
        }

        /// If nothing works we decode the class name from the typeid.
        return sofa::helper::NameDecoder::decodeClassName(typeid(T));
    }

    template<class T>
    static const std::string getOverridableTemplateName()
    {
        /// If the T object implement a GetCustomTemplateName static method... then we
        /// use that to return the name.
        if constexpr (HasGetCustomTemplateName<T>::value)
                return T::GetCustomTemplateName();

        //TODO(dmarchal 01/04/2020): compatibility layer, remove after 01/01/2020
        // use the templateName method.
        if constexpr (HasDeprecatedTemplateName<T>::value)
        {
            T* ptr {nullptr};
            const std::string n = T::templateName(ptr);
            return n;
        }

        /// A GetDefaultTemplateName method is added by the SOFA_CLASS macro, if the object is
        /// has not better option it will use this one.
        if constexpr (HasGetDefaultTemplateName<T>::value)
                return T::GetDefaultTemplateName();

        /// Finally if nothing matches, decode the template name from the typeid
        return sofa::helper::NameDecoder::decodeTemplateName(typeid(T));
    }
};

template<typename T>
std::string GetSofaTypeTemplateName(const std::string prefix)
{
    if constexpr (HasName<T>::value )
            return prefix + T::Name();
    else if constexpr (HasDataTypeInfo<T>::value )
            return prefix + sofa::defaulttype::DataTypeInfo<T>::name();
    else
        return prefix + sofa::helper::NameDecoder::decodeTypeName(typeid(T));
}

template<typename T1, typename T2, typename ...Ts>
std::string GetSofaTypeTemplateName(const std::string prefix)
{
    return GetSofaTypeTemplateName<T1>(prefix) + GetSofaTypeTemplateName<T2, Ts...>(",");
}

} // namespace sofa::core::objectmodel


