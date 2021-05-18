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
#include <sofa/defaulttype/TypeInfoID.h>
#include <vector>
#include <string>

namespace sofa::defaulttype
{

//////////////////////////////// Forward declaration //////////////////////////////
class AbstractTypeInfo;
///////////////////////////////////////////////////////////////////////////////////

/** *******************************************************************************
 * @brief An unique singleton to register all the type info defined in Sofa
 *
 * AbstractTypeInfo offers an API to manipulate the data content of a specific type
 * without requiering the inner details of the type. Have a look in AbstractTypeInfo
 * for more informations.
 *
 * On its side, TypeInfoRegistry holds all the instances of object thats inherits
 * from AbstractTypeInfo
 *
 * The common use case is get the type id to access a full AbstractTypeInfo from
 * the TypeInfoRegistry. The acces is done with a TypeInfoId instance that stores
 * an unique identifier for each data type.
 *
 * Example of use:
 *      TypeInfoId& shortinfo = TypeInfoId::GetTypeId<double>();
 *      AbstractTypeInfo* info = TypeInfoRegistry::Get(shortinfo);
 *      info->getName()
 **********************************************************************************/
class SOFA_DEFAULTTYPE_API TypeInfoRegistry
{
public:
    /// Returns the abstractTypeInfo corresponding to the provided TypeInfoId
    /// If there is none a NamedOnlyTypeInfo object is created an returned
    static const AbstractTypeInfo* Get(const TypeInfoId& id);

    /// Register a new AbstractTypeInfo to the provided TypeInfoId. A Third parameter is used to
    /// provides the compilationTarget where the typeinfo is declared to ease the tracking of DataTypes.
    static int Set(const TypeInfoId& tid, AbstractTypeInfo* info, const std::string& compilationTarget);

    /// Returns a vecotr with all the AbstractTypeInfo that have been registered in the specified target.
    /// An empty target select everything that is in the registry.
    static std::vector<const AbstractTypeInfo*> GetRegisteredTypes(const std::string& target="");

    /// Returns a new int to generates the corresponding TypeInfoId.
    static int AllocateNewTypeId(const std::type_info& nfo);
};

}
