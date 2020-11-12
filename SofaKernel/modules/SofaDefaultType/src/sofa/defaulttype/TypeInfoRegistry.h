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
#include <string>

namespace sofa::defaulttype
{

//////////////////////////////// Forward declaration //////////////////////////////
class AbstractTypeInfo;
class TypeInfoId;
///////////////////////////////////////////////////////////////////////////////////

/** *******************************************************************************
 * @brief A unique singleton to register all the type info defined in Sofa
 *
 * The common use case is get the type id to access a full AbstractTypeInfo from
 * the TypeInfoRegistry.
 * Example:
 *      TypeInfoId& shortinfo = TypeInfoId::getTypeId<double>();
 *      AbstractTypeInfo* info = TypeInfoRegistry::Get(shortinfo.id);
 *      info->getName()
 **********************************************************************************/
class TypeInfoRegistry
{
public:
    static std::vector<const AbstractTypeInfo*> GetRegisteredTypes(const std::string& target="");
    static const AbstractTypeInfo* Get(const TypeInfoId& id);
    static int Set(const TypeInfoId& tid, AbstractTypeInfo* info, const std::string& compilationTarget);
};

enum class TypeInfoType
{
    COMPLETE,
    PARTIAL,
    ALL
};


}
