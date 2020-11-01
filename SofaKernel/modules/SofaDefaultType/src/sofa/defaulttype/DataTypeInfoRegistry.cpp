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
#include <sofa/defaulttype/DataTypeInfoRegistry.h>
#include <typeindex>
#include <map>
#include <iostream>
#include <sofa/defaulttype/AbstractTypeInfo.h>
namespace sofa::defaulttype
{

static std::map<const int, const AbstractTypeInfo*>& getMap()
{
    static std::map<const int, const AbstractTypeInfo*> typeinfos;
    return typeinfos;
}

const AbstractTypeInfo* DataTypeInfoRegistry::Get(const int id)
{
    auto& typeinfos = getMap();
    if( typeinfos.find(id) != typeinfos.end() )
        return typeinfos[id];

    std::cout << "WARNING WARNING... searching for type '"<< id << "' the type is not there...what can we do" << std::endl;
    return nullptr;
}

int DataTypeInfoRegistry::Set(const int id, AbstractTypeInfo* info)
{
    auto& typeinfos = getMap();
    if( typeinfos.find(id) != typeinfos.end() )
    {
        if( typeinfos[id] != info && info->ValidInfo())
        {
            std::cout << " Promoting typeinfo "<< id << " from " << typeinfos[id]->name() << " to " << info->name() << std::endl;
            typeinfos[id] = info;
            return 2;
        }
        return -1;
    }
    std::cout << " Registering a new type info at "  << id << " => " << info->name() <<std::endl;
    typeinfos[id] = info;
    return 1;
}


}
