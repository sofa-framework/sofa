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
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/helper/NameDecoder.h>
#include <sofa/helper/BackTrace.h>
#include <algorithm>
namespace sofa::defaulttype
{

static std::vector<const AbstractTypeInfo*>& getStorage()
{
    static std::vector<const AbstractTypeInfo*> typeinfos;
    return typeinfos;
}

std::vector<const AbstractTypeInfo*> DataTypeInfoRegistry::GetRegisteredTypes(const std::string& target)
{
    std::vector<const AbstractTypeInfo*> tmp;
    for(auto info : getStorage())
    {
        if(info->getCompilationTarget() == target)
        tmp.push_back(info);
    }
    return tmp;
}

const AbstractTypeInfo* DataTypeInfoRegistry::Get(const BaseDataTypeId& tid)
{
    size_t id = tid.id;
    auto& typeinfos = getStorage();

    if( id < typeinfos.size() && typeinfos[id] != nullptr )
        return typeinfos[id];

    std::cout << "Missing type '"<< id << "' the type is not there..." << std::endl;
    std::cout << "     name: " << sofa::helper::NameDecoder::decodeFullName(tid.nfo) << std::endl;
    auto stacktrace = sofa::helper::BackTrace::getTrace(5);
    for(size_t i=1;i<stacktrace.size();i++)
        std::cout << "    ["<<i<< "]" << stacktrace[i] << std::endl;

    return nullptr;
}

int DataTypeInfoRegistry::Set(const BaseDataTypeId& tid, AbstractTypeInfo* info, const std::string &compilationTarget)
{
    auto& typeinfos = getStorage();
    size_t id = tid.id;
    info->setCompilationTarget(compilationTarget);

    if( id >= typeinfos.size() )
    {
        typeinfos.resize(id+1, nullptr);
    }

    if( typeinfos[id] != nullptr )
    {
        if( typeinfos[id] != info && info->ValidInfo())
        {
            std::cout << " Promoting typeinfo "<< id << " from " << typeinfos[id]->getName() << " to " << info->getName() << std::endl;
            info->setCompilationTarget(compilationTarget);
            typeinfos[id] = info;
            return 2;
        }
        return -1;
    }
    if( info->ValidInfo() )
    {
        std::cout << " Registering a complete type info at "  << id << " => " << info->getName() << " - " << info->getTypeName() << std::endl;
    }
    else
    {
        std::cout << " Registering a partial new type info at "  << id << " => " << info->getName() << std::endl;

        auto stacktrace = sofa::helper::BackTrace::getTrace(10);
        for(size_t i=1;i<stacktrace.size();i++)
            std::cout << "    ["<<i<< "]" << stacktrace[i] << std::endl;

    }
    info->setCompilationTarget(compilationTarget);
    typeinfos[id] = info;
    return 1;
}


}
