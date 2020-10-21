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

std::map<const std::type_index, const AbstractTypeInfo*> typeinfos;

const AbstractTypeInfo* DataTypeInfoRegistry::Get(const std::type_info& id)
{
    auto index = std::type_index(id);
    if( typeinfos.find(index) != typeinfos.end() )
        return typeinfos[index];

    std::cout << "WARNING WARNING... searching for type '"<< id.name() << "' the type is not there...what can we do" << std::endl;
    return nullptr;
}

int DataTypeInfoRegistry::Set(const std::type_info& id, AbstractTypeInfo* info)
{
    auto index = std::type_index(id);
    std::cout << " REGISTER A NEW TYPE FOR "  << id.name() << "=>" << info->name() <<std::endl;
    if( typeinfos.find(index) != typeinfos.end() )
    {
        if( typeinfos[index] != info )
        {
            return 2;
        }
        return -1;
    }
    typeinfos[index] = info;
    return 1;
}


}
