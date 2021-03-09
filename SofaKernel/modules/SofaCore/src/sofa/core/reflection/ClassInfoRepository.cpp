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

#include <algorithm>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/NameDecoder.h>
#include "ClassInfo.h"
#include "ClassId.h"
#include "ClassInfoRepository.h"
#include <mutex>
namespace sofa::core::reflection
{

namespace {
    std::mutex ClassInfoMutex;
};

/// Private function returning the storage of all the ClassInfo instances.
/// ClassInfo instances are stored in a linear storage so we can access them in constant time
/// with an index.
static std::vector<const ClassInfo*>& getStorage()
{
    static std::vector<const ClassInfo*> typeinfos {};
    return typeinfos;
}

/// Private function returning the index of all the ClassInfo instances.
/// This one allowed, at initialization to detect in non-constant time which ClassInfo
/// instances are already registered and if so, at which index.
static std::map<std::type_index, int>& getIndex()
{
    static std::map<std::type_index, int> index{};
    return index;
}

/// Returns a vector of all registered type matching a given compilation target.
std::vector<const ClassInfo*> ClassInfoRepository::GetRegisteredTypes(const std::string& target)
{
    std::scoped_lock autolock(ClassInfoMutex);

    bool selectAll = target == "";
    std::vector<const ClassInfo*> tmp;
    for(auto info : getStorage())
    {
        if(info==nullptr)
            continue;

        if(selectAll || info->compilationTarget == target)
            tmp.push_back(info);
    }
    return tmp;
}

bool ClassInfoRepository::HasACompleteEntryFor(const ClassId& tid)
{
    std::scoped_lock autolock(ClassInfoMutex);

    sofa::Size id = tid.id;
    auto& typeinfos = getStorage();

    return (id < typeinfos.size() && typeinfos[id] != nullptr);
}


const ClassInfo* ClassInfoRepository::Get(const ClassId& tid)
{
    std::scoped_lock autolock(ClassInfoMutex);

    sofa::Size id = tid.id;
    auto& typeinfos = getStorage();

    if( id < typeinfos.size() && typeinfos[id] != nullptr)
        return typeinfos[id];

    std::stringstream tmp;
    tmp << "Accessing an invalid class info for type " << tid.symbol.name();
    throw std::runtime_error(tmp.str().c_str());
}

int ClassInfoRepository::AllocateNewTypeId(const std::type_index& tindex)
{
    std::scoped_lock autolock(ClassInfoMutex);

    auto& index = getIndex();
    auto fi = index.find(tindex);
 
    /// if the nfo index is not yet registered then we create a new entry. 
    if (fi == index.end())
    {
        auto& typeinfos = getStorage();
        typeinfos.push_back(nullptr);
        int newindex = typeinfos.size() - 1;
        index[tindex] = newindex;
        return newindex;
    }
    return fi->second;
}

int ClassInfoRepository::Set(const ClassId& tid, const ClassInfo* info)
{
    std::scoped_lock autolock(ClassInfoMutex);

    /// If there is no typeinfo to register, returns
    if( info == nullptr )
        return -1;

    auto& typeinfos = getStorage();
    sofa::Index id = tid.id;

    /// If the current storage for classinfo is too small, then increase its
    /// size and fill it with NoClassInfo::GetInstance();
    if( id >= typeinfos.size() )
    {
        typeinfos.resize(id+1, nullptr);
    }

    /// If the current storage at 'id' return something and that this is not of type NoClassInfo
    /// we should check if we are not overriding an existing and valid values (this should not
    /// happens as it means we are duplicating the registration).
    if( typeinfos[id] )
    {
        /// If the existing pointer and the new one are different, we override the old value
        /// print a error message and returns.
        if( typeinfos[id] != info)
        {
            dmsg_error("ClassInfoRegistry") << " Overriding a typeinfo "<< id << " from " << typeinfos[id]->className << " to " << info->className;
            typeinfos[id] = info;
            return 2;
        }

        /// We are registering the same class info two time.
        return -1;
    }

    /// We are registering the class info for the first time.
    typeinfos[id] = info;
    return 1;
}


}
