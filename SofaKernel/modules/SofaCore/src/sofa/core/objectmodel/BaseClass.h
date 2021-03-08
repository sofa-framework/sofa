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
#include <sofa/core/fwd.h>
#include <tuple>
#include <sofa/core/objectmodel/SPtr.h>
#include <sofa/helper/NameDecoder.h>
#include <sofa/core/reflection/Class.h>

namespace sofa::core::objectmodel
{

#define classid(T) *sofa::core::reflection::Class::GetClassInfo<T>()

/// To specify template classes in C macro parameters, we can't write any commas, hence templates with more than 2 parameters have to use the following macros
#define SOFA_TEMPLATE(Class,P1) Class<P1>
#define SOFA_TEMPLATE2(Class,P1,P2) Class<P1,P2>
#define SOFA_TEMPLATE3(Class,P1,P2,P3) Class<P1,P2,P3>
#define SOFA_TEMPLATE4(Class,P1,P2,P3,P4) Class<P1,P2,P3,P4>

// This macro should now be used to declare the Base class as part of the reflection system.
#define SOFA_BASE_CLASS(T) \
    typedef void ParentClasses; \
    static const ::sofa::core::objectmodel::BaseClass* GetClass() { return sofa::core::reflection::Class::GetClassInfo<T>(); } \
    virtual const ::sofa::core::objectmodel::BaseClass* getClass() const { return sofa::core::reflection::Class::GetClassInfo<T>(); } \

// This macro should now be used at the beginning of all declarations of classes with 1 base class
#define SOFA_CLASS(T,Parent) \
    typedef T MyType; \
    typedef std::tuple< Parent > ParentClasses; \
    typedef Parent Inherit1; \
    SOFA_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 1 base class
#define SOFA_ABSTRACT_CLASS(T,Parent) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent > ParentClasses; \
    typedef Parent Inherit1; \
    SOFA_ABSTRACT_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 2 base classes
#define SOFA_CLASS2(T,Parent1,Parent2) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    SOFA_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 2 base classes
#define SOFA_ABSTRACT_CLASS2(T,Parent1,Parent2) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    SOFA_ABSTRACT_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 3 base classes
#define SOFA_CLASS3(T,Parent1,Parent2,Parent3) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    SOFA_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 3 base classes
#define SOFA_ABSTRACT_CLASS3(T,Parent1,Parent2,Parent3) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    SOFA_ABSTRACT_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 4 base classes
#define SOFA_CLASS4(T,Parent1,Parent2,Parent3,Parent4) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3, Parent4 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    typedef Parent4 Inherit4; \
    SOFA_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 4 base classes
#define SOFA_ABSTRACT_CLASS4(T,Parent1,Parent2,Parent3,Parent4) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3, Parent4 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    typedef Parent4 Inherit4; \
    SOFA_ABSTRACT_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 5 base classes
#define SOFA_CLASS5(T,Parent1,Parent2,Parent3,Parent4,Parent5) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3, Parent4, Parent5 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    typedef Parent4 Inherit4; \
    typedef Parent5 Inherit5; \
    SOFA_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 5 base classes
#define SOFA_ABSTRACT_CLASS5(T,Parent1,Parent2,Parent3,Parent4,Parent5) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3, Parent4, Parent5 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    typedef Parent4 Inherit4; \
    typedef Parent5 Inherit5; \
    SOFA_ABSTRACT_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 5 base classes
#define SOFA_CLASS6(T,Parent1,Parent2,Parent3,Parent4,Parent5,Parent6) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3, Parent4, Parent5, Parent6 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    typedef Parent4 Inherit4; \
    typedef Parent5 Inherit5; \
    typedef Parent6 Inherit6; \
    SOFA_CLASS_DECL

// This macro should now be used at the beginning of all declarations of classes with 5 base classes
#define SOFA_ABSTRACT_CLASS6(T,Parent1,Parent2,Parent3,Parent4,Parent5,Parent6) \
    typedef T MyType;                                               \
    typedef std::tuple< Parent1, Parent2, Parent3, Parent4, Parent5, Parent6 > ParentClasses; \
    typedef Parent1 Inherit1; \
    typedef Parent2 Inherit2; \
    typedef Parent3 Inherit3; \
    typedef Parent4 Inherit4; \
    typedef Parent5 Inherit5; \
    typedef Parent6 Inherit6; \
    SOFA_ABSTRACT_CLASS_DECL

// Do not use this macro directly, use SOFA_ABSTRACT_CLASS instead
#define SOFA_ABSTRACT_CLASS_DECL                                        \
    using Ptr = MyType*;                                                \
    using SPtr = sofa::core::sptr<MyType>;                              \
    friend class sofa::helper::NameDecoder;                             \
    static std::string GetDefaultTemplateName(){ return sofa::helper::NameDecoder::DefaultTypeTemplateName<MyType>::Get(); } \
    static const ::sofa::core::objectmodel::BaseClass* GetClass() { return sofa::core::reflection::Class::GetClassInfo<MyType>(); }   \
    const ::sofa::core::objectmodel::BaseClass* getClass() const override { return sofa::core::reflection::Class::GetClassInfo<MyType>(); }  \
    static const char* HeaderFileLocation() { return __FILE__; }        \
    template<class SOFA_T> ::sofa::core::objectmodel::BaseData::BaseInitData \
    initData(::sofa::core::objectmodel::Data<SOFA_T>* field, const char* name, const char* help,   \
    ::sofa::core::objectmodel::BaseData::DataFlags dataflags)  \
{                                                                   \
    ::sofa::core::objectmodel::BaseData::BaseInitData res;          \
    this->initData0(field, res, name, help, dataflags);             \
    return res;                                                     \
}                                                                   \
    template<class SOFA_T> ::sofa::core::objectmodel::BaseData::BaseInitData \
    initData(::sofa::core::objectmodel::Data<SOFA_T>* field, const char* name, const char* help,   \
    bool isDisplayed=true, bool isReadOnly=false)              \
{                                                                   \
    ::sofa::core::objectmodel::BaseData::BaseInitData res;          \
    this->initData0(field, res, name, help,                         \
    isDisplayed, isReadOnly);                       \
    return res;                                                     \
}                                                                   \
    template<class SOFA_T> typename ::sofa::core::objectmodel::Data<SOFA_T>::InitData initData(    \
    ::sofa::core::objectmodel::Data<SOFA_T>* field, const SOFA_T& value, const char* name,     \
    const char* help, bool isDisplayed=true, bool isReadOnly=false) \
{                                                                   \
    typename ::sofa::core::objectmodel::Data<SOFA_T>::InitData res; \
    this->initData0(field, res, value, name, help,                  \
    isDisplayed, isReadOnly);                       \
    return res;                                                     \
}                                                                   \
    ::sofa::core::objectmodel::BaseLink::InitLink<MyType>               \
    initLink(const char* name, const char* help)                        \
{                                                                   \
    return ::sofa::core::objectmodel::BaseLink::InitLink<MyType>    \
    (this, name, help);                                             \
}                                                                   \
    using Inherit1::sout;                                               \
    using Inherit1::serr;                                               \
    using Inherit1::sendl

// Do not use this macro directly, use SOFA_CLASS instead
#define SOFA_CLASS_DECL                                        \
    SOFA_ABSTRACT_CLASS_DECL;                                  \
    \
    friend class sofa::core::objectmodel::New<MyType>

} // namespace sofa




