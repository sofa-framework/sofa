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

#include <sofa/core/reflection/ClassInfo.h>
#include <sofa/core/reflection/ClassId.h>
#include <sofa/core/reflection/ClassInfoBaseImpl.h>
#include <sofa/core/reflection/ClassInfoRepository.h>
#include <sofa/helper/NameDecoder.h>
#include <tuple>
#include <iostream>

using sofa::helper::NameDecoder;

namespace sofa::core::reflection
{
/** ************************************************************************
 * @brief Generates an unique instance of ClassInfo for the type parameter T.
 *
 * The common use case is get the type id to access a full AbstractTypeInfo from
 * the TypeInfoRegistry.
 * Example:
 *    ClassInfoBuilder::GetClassInfo<MyTypeInheritingFromBase>();
 *****************************************************************************/
class ClassInfoBuilder
{
public:
    /// Get the class info associated constructed from the type parameter T.
    /// The instance is unique and every call should returns the same value.
    template<class T>
    static const ClassInfo* GetOrBuildClassInfo(const std::string& compilationTarget)
    {
        static const ClassInfo* info = CreateNewInstance<T>(Class::GetClassId<T>(), compilationTarget);
        return info;
    }

private:
    /// Internal implementation that actually get or create an instance for the given type as well
    /// as its full in-heritance tree.
    template<class T>
    static const ClassInfo* CreateNewInstance(const ClassId& cid, const std::string& compilationTarget);
};

/// Private namespace because we don't want that to be used by the user of the API.
namespace
{
/// Recursive template to generates the class info all along the in-heritance tree
/// The recursion continue as long as there is a parent of non void type.
template<class Tuple>
class Parents
{
public:
    constexpr static int Nb() { return std::tuple_size<Tuple>::value; }

    static const ClassInfo* Get(const int i, const std::string& compilationTarget)
    {
        return GetRec<Nb()-1>(i, compilationTarget);
    }

    template<int j>
    static const ClassInfo* GetRec(int i, const std::string& compilationTarget)
    {
        if(i==j)
        {
            return ClassInfoBuilder::GetOrBuildClassInfo<typename std::template tuple_element<j,Tuple>::type>(compilationTarget);
        }
        if constexpr (j>0)
                return GetRec<j-1>(i, compilationTarget);
        return nullptr;
    }
};

/// Recursive template to generates the class info all along the in-heritance tree
/// Termination Case is when the Parents is of type void.
template<>
class Parents<void>
{
public:
    constexpr static int Nb(){ return 0; }
    constexpr static const ClassInfo* Get(int, const std::string&){ return nullptr; }
};
}

/// Create method that generates a complete the class info for the complete
/// in-heritance tree.
///
/// - If a ClassInfo already exists for a given parent type then this one is used
///   instead of recreating a new one.
/// - The function use NameDecoder to compute appropriate names
///
template<class T>
const ClassInfo* ClassInfoBuilder::CreateNewInstance(const ClassId& cid, const std::string& compilationTarget)
{
    if (ClassInfoRepository::HasACompleteEntryFor(cid))
        return ClassInfoRepository::Get(cid);

    ClassInfo* newinfo = new ClassInfoBaseImpl<T>();
    newinfo->typeName = NameDecoder::getTypeName<T>();
    newinfo->namespaceName = NameDecoder::getNamespaceName<T>();
    newinfo->className = NameDecoder::getClassName<T>();
    newinfo->templateName = NameDecoder::getTemplateName<T>();
    newinfo->shortName = NameDecoder::getShortName<T>();
    newinfo->compilationTarget = compilationTarget;

    newinfo->parents.resize(Parents<typename T::ParentClasses>::Nb());
    for (unsigned int i = 0; i < newinfo->parents.size(); i++)
    {
        newinfo->parents[i] = Parents<typename T::ParentClasses>::Get(i, compilationTarget);
    }
    ClassInfoRepository::Set(Class::GetClassId<T>(), newinfo);
    return newinfo;
}

} /// sofa::core::objectmodel
