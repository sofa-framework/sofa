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
#include <sofa/core/reflection/ClassId.h>
#include <sofa/core/fwd.h>
#include <vector>
#include <string>

namespace sofa::core::reflection
{

/** *******************************************************************************
 * @brief An unique singleton to register all the type info defined in Sofa
 *
 * ClassInfo offers an API to manipulate the data content of a specific type
 * without requiering the inner details of the type. Have a look in AbstractTypeInfo
 * for more informations.
 *
 * On its side, ClassInfoRegistry holds all the instances of object thats inherits
 * from ClassInfo
 *
 * The common use case is get the type id to access a full ClassInfo from
 * the ClassInfoRegistry. The acces is done with a ClassId instance that stores
 * an unique identifier for each data type.
 *
 * Example of use:
 *      ClassId& shortinfo = ClassId::GetTypeId<double>();
 *      ClassInfo* info = ClassInfoRegistry::Get(shortinfo);
 *      info->getName()
 **********************************************************************************/
class SOFA_CORE_API ClassInfoRepository
{
public:
    /// Returns true if a complete class info is available, false otherwise 
    /// A classinfo could be fully completed or just storing "MissingTypeInfo".
    static bool HasACompleteEntryFor(const ClassId& id);

    /// Returns the abstractTypeInfo corresponding to the provided TypeInfoId
    /// If there is none a NamedOnlyTypeInfo object is created an returned
    static const ClassInfo* Get(const ClassId& id);

    /// Register a new AbstractTypeInfo to the provided TypeInfoId. A Third parameter is used to
    /// provides the compilationTarget where the typeinfo is declared to ease the tracking of DataTypes.
    static int Set(const ClassId& tid, const ClassInfo* info);

    /// Returns a vector with all the AbstractTypeInfo that have been registered in the specified target.
    /// An empty target select everything that is in the registry.
    static std::vector<const ClassInfo*> GetRegisteredTypes(const std::string& target="");

    /// Returns a new int to generates the corresponding TypeInfoId.
    static int AllocateNewTypeId(const std::string& nfo);
};

}
