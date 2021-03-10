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
#include <sofa/helper/TypeInfo.h>
#include <vector>
#include <string>
namespace sofa::core::reflection
{

namespace {
    using sofa::core::objectmodel::Base;
}

/**
 *  \brief Class info reflection class
 *
 *  This class provides information on the class and parent classes of components.
 *
 *  Examples:
 *      ClassInfo *info = ClassId::ClassId::GetClassId<MyObject>()->getClassInfo();
 *      info->className
 *      info->isInstance(otherObjectInfo);
 **/
class SOFA_CORE_API ClassInfo
{
private:
    const std::type_info* pt;

protected:
    ClassInfo(const std::type_info* ti) { pt = ti; }
    virtual ~ClassInfo(){}

public:
    /// The following was from BaseClass
    std::string compilationTarget;       ///< In which SOFA_TARGET is registered this type
    std::string namespaceName;           ///< The c++ namespace
    std::string typeName;                ///< The c++ typename
    std::string className;               ///< The 'sofa' object class name (can be customized)
    std::string templateName;            ///< The 'sofa' object's template name (can be customized)
    std::string shortName;               ///< A Short name for the class
    std::vector<const ClassInfo*> parents; ///< The list of the the c++ parents class

    /// The following was from ClassInfo (to deprecate ?)
    sofa::helper::TypeInfo type() const { return sofa::helper::TypeInfo(*pt); }

    /// The following was from ClassInfo (to deprecate ?)
    const std::string& name() const { return className; }

    /// returns true iff c is a parent class of this
    virtual bool hasParent(const ClassInfo* c) const;

    /// returns true iff a parent class of this is named parentClassName
    virtual bool hasParent(const std::string& parentClassName) const;

    /// dynamic_casts the parameter 'obj' to the type this ClassInfo is describing
    /// and returns its as Base*.
    /// returns:
    ///   - nullptr if the cast fails.
    ///   - the same address as the parameter 'obj'
    virtual Base* dynamicCastToBase(Base* obj) const = 0;

    /// dynamic_casts the parameter 'obj' to the type this ClassInfo is describing
    /// returns:
    ///   - nullptr if the cast fails.
    ///   - the address of the casted pointer.
    ///
    /// Warning, the returned address may not be the same as the one from 'obj'.
    virtual void* dynamicCast(Base* obj) const = 0;

    /// Returns true if the parameter 'obj' is of same type as the type described
    /// by this ClassInfo
    virtual bool isInstance(Base* obj) const = 0;
};

} ///sofa::core::objectmodel
