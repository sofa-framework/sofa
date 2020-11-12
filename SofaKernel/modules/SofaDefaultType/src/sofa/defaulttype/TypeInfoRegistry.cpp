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
#include <typeindex>
#include <map>
#include <iostream>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/defaulttype/AbstractTypeInfo.h>
#include <sofa/defaulttype/DataTypeInfo.h>
#include <sofa/helper/NameDecoder.h>
#include <algorithm>
#include "TypeInfoID.h"
#include "TypeInfoRegistry.h"

namespace sofa::defaulttype
{

static std::vector<const AbstractTypeInfo*>& getStorage()
{
    static std::vector<const AbstractTypeInfo*> typeinfos;
    return typeinfos;
}

std::vector<const AbstractTypeInfo*> TypeInfoRegistry::GetRegisteredTypes(const std::string& target)
{
    std::vector<const AbstractTypeInfo*> tmp;
    for(auto info : getStorage())
    {
        if(info->getCompilationTarget() == target)
        tmp.push_back(info);
    }
    return tmp;
}

const AbstractTypeInfo* TypeInfoRegistry::Get(const TypeInfoId& tid)
{
    size_t id = tid.id;
    auto& typeinfos = getStorage();

    if( id < typeinfos.size() && typeinfos[id] != nullptr )
        return typeinfos[id];

    msg_error("TypeInfoRegistry") << "Missing type '"<< id << "' the type is not there..." << msgendl
                                  << "     name: " << sofa::helper::NameDecoder::decodeFullName(tid.nfo);

    return nullptr;
}

int TypeInfoRegistry::Set(const TypeInfoId& tid, AbstractTypeInfo* info, const std::string &compilationTarget)
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
            msg_info("TypeInfoRegistry") << " Promoting typeinfo "<< id << " from " << typeinfos[id]->name() << " to " << info->name();
            info->setCompilationTarget(compilationTarget);
            typeinfos[id] = info;
            return 2;
        }
        return -1;
    }
    if( info->ValidInfo() )
    {
        msg_info("TypeInfoRegistry") << " Registering a complete type info at "  << id << " => " << info->name();
    }
    else
    {
        msg_warning("TypeInfoRegistry") << " Registering a partial new type info at "  << id << " => " << info->name();
    }
    info->setCompilationTarget(compilationTarget);
    typeinfos[id] = info;
    return 1;
}


}
