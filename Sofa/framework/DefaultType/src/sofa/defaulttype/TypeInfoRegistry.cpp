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
#include <sofa/defaulttype/typeinfo/NoTypeInfo.h>
#include <sofa/defaulttype/typeinfo/NameOnlyTypeInfo.h>
#include <sofa/defaulttype/typeinfo/models/IncompleteTypeInfo.h>
#include <sofa/defaulttype/typeinfo/DataTypeInfoDynamicWrapper.h>

#include <algorithm>
#include <sofa/defaulttype/TypeInfoID.h>
#include <sofa/defaulttype/TypeInfoRegistry.h>

namespace sofa::defaulttype
{

static std::vector<const AbstractTypeInfo*>& getStorage()
{
    static std::vector<const AbstractTypeInfo*> typeinfos {NoTypeInfo::Get()};
    return typeinfos;
}

std::vector<const AbstractTypeInfo*> TypeInfoRegistry::GetRegisteredTypes(const std::string& target)
{
    const bool selectAll = target == "";
    std::vector<const AbstractTypeInfo*> tmp;
    for(auto info : getStorage())
    {
        if(info==nullptr)
            continue;

        if(selectAll || info->getCompilationTarget() == target)
            tmp.push_back(info);
    }
    return tmp;
}

const AbstractTypeInfo* TypeInfoRegistry::Get(const TypeInfoId& tid)
{
    const sofa::Size id = tid.id;
    const auto& typeinfos = getStorage();

    if( id < typeinfos.size() && typeinfos[id] != nullptr)
        return typeinfos[id];

    msg_error("TypeInfoRegistry") << "Missing typeinfo for '"<< sofa::helper::NameDecoder::decodeFullName(tid.nfo)
                                  << "' (searching at index " << tid.id  << ")";

    return nullptr;
}

int TypeInfoRegistry::AllocateNewTypeId(const std::type_info& nfo)
{
    auto& typeinfos = getStorage();
    const std::string name = sofa::helper::NameDecoder::decodeTypeName(nfo);
    const std::string typeName = sofa::helper::NameDecoder::decodeTypeName(nfo);
    typeinfos.push_back(new NameOnlyTypeInfo(name, typeName));
    return typeinfos.size()-1;
}


int TypeInfoRegistry::Set(const TypeInfoId& tid, AbstractTypeInfo* info, const std::string &compilationTarget)
{
    if( info == nullptr )
        return -1;

    auto& typeinfos = getStorage();
    const sofa::Size id = tid.id;

    msg_info("TypeInfoRegistry") << " Trying to register '"<< info->name() << "/" << tid.nfo.name() << "' at index " << id << "";

    info->setCompilationTarget(compilationTarget);
    if( id >= typeinfos.size() )
    {
        typeinfos.resize(id+1, NoTypeInfo::Get());
    }

    if( typeinfos[id] )
    {
        if( typeinfos[id] != info )
        {
            if( (typeinfos[id] == NoTypeInfo::Get()) || !typeinfos[id]->ValidInfo())
            {
                msg_info("TypeInfoRegistry") << " Promoting typeinfo "<< id << " from " << typeinfos[id]->name() << " to " << info->name();
                info->setCompilationTarget(compilationTarget);
                typeinfos[id] = info;
                return 2;
            }
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
    typeinfos[id] = info;
    return 1;
}


}
