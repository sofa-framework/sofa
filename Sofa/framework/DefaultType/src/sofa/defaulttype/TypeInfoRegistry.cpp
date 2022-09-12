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

static std::map<std::type_index, int>& getAccessMap()
{
    static std::map<std::type_index, int> mapToIndex;
    return mapToIndex;
}

static std::vector<const AbstractTypeInfo*>& getStorage()
{
    static std::vector<const AbstractTypeInfo*> typeinfos {NoTypeInfo::Get()};
    return typeinfos;
}

std::vector<const AbstractTypeInfo*> TypeInfoRegistry::GetRegisteredTypes(const std::string& target)
{
    bool selectAll = target == "";
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

// Get in the registry if a type already exists for this TypeInfoId
const AbstractTypeInfo* TypeInfoRegistry::Get(const TypeInfoId& tid)
{
    sofa::Size id = tid.id;
    auto& typeinfos = getStorage();
    if( id < typeinfos.size() && typeinfos[id] != nullptr)
        return typeinfos[id];

    dmsg_error("TypeInfoRegistry") << "Missing typeinfo for '"<< sofa::helper::NameDecoder::decodeFullName(tid.nfo)
                                   << "' (searching at index " << tid.id  << ")";

    return nullptr;
}

// This function has a non linear comlpexity but it should be called only once per static allocation of a type id. 
int TypeInfoRegistry::AllocateNewTypeId(const std::type_info& nfo)
{
    auto& typeinfos = getStorage();
    auto& map = getAccessMap();
    
    // search in the map if there is not a type already an entry for that type_info
    auto nfoInMap = map.find(std::type_index(nfo));
    if( nfoInMap != map.end() )
    {
        // there is a match if a TypeInfoId has already been allocated for that type_info. 
        // this can happens when TypeInfoId::Get is called for the same type in multiple different shared libraries.
        // to be sure there is only one single "int" entry for a type we have to perform this check. 
        dmsg_info("TypeInfoRegistry") << " Trying to register '"<< typeinfos[nfoInMap->second]->name() << " registered in " << typeinfos[nfoInMap->second]->getCompilationTarget();        
        return nfoInMap->second;
    }
    
    // create the name & typename 
    std::string name = sofa::helper::NameDecoder::decodeTypeName(nfo);
    std::string typeName = sofa::helper::NameDecoder::decodeTypeName(nfo);
    
    // register a minimal type "name only" type info. 
    typeinfos.push_back(new NameOnlyTypeInfo(name, typeName));
    
    // get the type id
    int id = typeinfos.size()-1;
    
    // store it in the access map for future checking of existence. 
    map[std::type_index(nfo)] = 0;    
    return id;
}

int TypeInfoRegistry::Set(const TypeInfoId& tid, AbstractTypeInfo* info, const std::string &compilationTarget)
{
    if( info == nullptr )
        return -1;

    auto& typeinfos = getStorage();
    sofa::Size id = tid.id;

    dmsg_info("TypeInfoRegistry") << " Trying to register '"<< info->name() << "/" << tid.nfo.name() << "' at index " << id << "";

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
                dmsg_info("TypeInfoRegistry") << " Promoting typeinfo "<< id << " from " << typeinfos[id]->name() << " to " << info->name();
                info->setCompilationTarget(compilationTarget);
                typeinfos[id] = info;
                return 2;
            }
        }
        return -1;
    }
    if( info->ValidInfo() )
    {
        dmsg_info("TypeInfoRegistry") << " Registering a complete type info at "  << id << " => " << info->name();
    }
    else
    {
        dmsg_warning("TypeInfoRegistry") << " Registering a partial new type info at "  << id << " => " << info->name();
    }
    typeinfos[id] = info;
    return 1;
}


}
