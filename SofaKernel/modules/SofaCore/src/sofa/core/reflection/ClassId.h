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
#include <sofa/core/reflection/fwd.h>
#include <type_traits>
#include <typeindex>
#include <string>

namespace sofa::core::reflection
{

/** ************************************************************************
 * @brief Generates unique id for object inirited from class.
 *
 * Compared to type_info.hash_code() this version is guaranteed to have an amortized
 * constant time. There is a one to one mapping between std::type_info and ClassId
 *
 * The common use case is get the type id to access a full AbstractTypeInfo from
 * the TypeInfoRegistry.
 * Example:
 *      ClassId& shortinfo = ClassId::getClassId<double>();
 *      ClassInfo* info = ClassInfoRegistry::Get(shortinfo);
 *      info->getName()
 *****************************************************************************/
class SOFA_CORE_API ClassId
{

public:
    /// Returns the ClassInfo associated with this ClassId.
    const sofa::core::reflection::ClassInfo* getClassInfo() const ;

    sofa::Index id;
    std::type_index symbol;

    ClassId(const std::type_info& s);
};

} /// namespace sofa::defaulttype
