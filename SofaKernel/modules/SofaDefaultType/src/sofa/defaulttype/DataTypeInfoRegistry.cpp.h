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
#include <string>

namespace sofa::defaulttype
{

class DataTypeInfoRegistry
{
public:

    template<class T>
    static const AbstractTypeInfo* Get(const T& o)
    {
        return Get(typeid(T));
    }


    static const AbstractTypeInfo* Get(const std::type_info& id)
    {
        auto index = std::type_index(id);
        if( typeinfos.find(index) != typeinfos.end() )
            return typeinfos[index];
        return nullptr;
    }

    static void registerTypeInfo(const std::type_info& id, AbstractTypeInfo* info)
    {
        auto index = std::type_index(id);
        std::cout << "Adding in the map" << id.name() << std::endl;
        if( typeinfos.find(index) != typeinfos.end() )
        {
            if( typeinfos[index] != info )
            {
                std::cout << "Trying to register type with a different abstract type info" << std::endl;
            }
            return;
        }
        typeinfos[index] = info;
        return;
    }

private:
    static std::map<const std::type_index, const AbstractTypeInfo*> typeinfos;
};

std::map<const std::type_index, const AbstractTypeInfo*> DataTypeInfo::typeinfos;


}
