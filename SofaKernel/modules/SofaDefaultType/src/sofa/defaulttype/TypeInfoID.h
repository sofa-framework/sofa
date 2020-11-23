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

#include <typeinfo>

namespace sofa::defaulttype
{

/** ************************************************************************
 * @brief Generates unique id for types.
 *
 * Compared to type_info.hash_code() this version is guaranteed to be in
 * constant time
 *
 * The common use case is get the type id to access a full AbstractTypeInfo from
 * the TypeInfoRegistry.
 * Example:
 *      TypeInfoId& shortinfo = TypeInfoId::getTypeId<double>();
 *      AbstractTypeInfo* info = TypeInfoRegistry::Get(shortinfo.id);
 *      info->getName()
 *****************************************************************************/
class SOFA_DEFAULTTYPE_API TypeInfoId
{
public:
    template<class T>
    static const TypeInfoId& GetTypeId()
    {
        static TypeInfoId typeId(TypeInfoId::GetNewId(typeid(T)), typeid(T));
        return typeId;
    }

    int id;
    const std::type_info& nfo;

private:
    TypeInfoId(int id_, const std::type_info& nfo);
    static int GetNewId(const std::type_info& nfo);
};

} /// namespace sofa::defaulttype
