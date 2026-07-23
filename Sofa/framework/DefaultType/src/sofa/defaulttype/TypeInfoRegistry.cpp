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
#include <memory>
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

/// Non-owning list of registered type infos, indexed by TypeInfoId.
/// Most entries are borrowed pointers to function-local static singletons
/// (e.g. NoTypeInfo::Get() or DataTypeInfoDynamicWrapper<...>::get()) that
/// must never be deleted here. The few objects actually allocated by this
/// registry are owned by getOwnedStorage() instead.
static std::vector<const AbstractTypeInfo*>& getStorage()
{
    static std::vector<const AbstractTypeInfo*> type_infos{NoTypeInfo::Get()};
    return type_infos;
}

/// Owns the type infos allocated by this registry itself (see AllocateNewTypeId).
/// The raw pointers stored in getStorage() alias these objects, so ownership is
/// kept here to release them at exit without deleting borrowed singletons.
static std::vector<std::unique_ptr<const AbstractTypeInfo>>& getOwnedStorage()
{
    static std::vector<std::unique_ptr<const AbstractTypeInfo>> owned_type_infos;
    return owned_type_infos;
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
    auto& owned = getOwnedStorage().emplace_back(std::make_unique<NameOnlyTypeInfo>(name, typeName));
    typeinfos.push_back(owned.get());
    return static_cast<int>(typeinfos.size()-1);
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
