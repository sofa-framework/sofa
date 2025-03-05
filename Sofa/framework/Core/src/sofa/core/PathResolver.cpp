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
#include <regex>
#include <sofa/core/PathResolver.h>
#include <sofa/core/objectmodel/DataLink.h>
#include <sofa/core/objectmodel/Base.h>

namespace sofa::core
{

using sofa::core::objectmodel::Base;
using sofa::core::objectmodel::BaseData;
using sofa::core::objectmodel::BaseClass;
using sofa::core::objectmodel::AbstractDataLink;

bool PathResolver::PathHasValidSyntax(const std::string &path)
{
    const std::regex forbiddenSymbols {"[^\\./[:alnum:]]"};
    return !std::regex_search(path, forbiddenSymbols);
}

Base* PathResolver::FindBaseFromPath(const Base* context, const std::string& path)
{
    if(context==nullptr)
        return nullptr;

    Base* b {nullptr};
    const_cast<Base*>(context)->findLinkDest(b, path, nullptr);
    return b;
}

Base* PathResolver::FindBaseFromClassAndPath(const Base* context, const BaseClass* tclass, const std::string& path)
{
    if(context==nullptr)
        return nullptr;

    if(tclass==nullptr)
        return nullptr;

    Base* b = const_cast<Base*>(context)->findLinkDestClass(tclass, path, nullptr);
    return b;
}

BaseData* PathResolver::FindBaseDataFromPath(const BaseData* datacontext, const std::string& path)
{
    if(datacontext==nullptr)
        return nullptr;

    Base* context = datacontext->getOwner();
    if(context==nullptr)
        return nullptr;

    BaseData* b {nullptr};
    const_cast<Base*>(context)->findDataLinkDest(b, path, nullptr);
    return b;
}

BaseData* PathResolver::FindBaseDataFromPath(Base* context, const std::string& path)
{
    if(context==nullptr)
        return nullptr;
    BaseData* b{nullptr};
    const_cast<Base*>(context)->findDataLinkDest(b, path, nullptr);
    return b;
}

bool PathResolver::FindDataLinkDest(Base* context, BaseData*& ptr, const std::string& path, const BaseLink* link)
{
    if(context==nullptr)
        return false;
    return context->findDataLinkDest(ptr, path, link);
}

Base* PathResolver::FindLinkDestClass(Base* context, const BaseClass* destType, const std::string& path, const BaseLink* link)
{
    if(context==nullptr)
        return nullptr;
    return context->findLinkDestClass(destType, path, link);
}

bool PathResolver::CheckPath(sofa::core::objectmodel::Base* context, const sofa::core::objectmodel::BaseClass* classType, const std::string& path)
{
    const void* tmp = PathResolver::FindLinkDestClass(context, classType, path, nullptr);
    return tmp != nullptr;
}

/// Check that a given list of path is valid, that the pointed object exists and is of the right type
bool PathResolver::CheckPaths(Base *context, const BaseClass* linktype,  const std::string& paths)
{
    if (paths.empty())
        return false;
    std::istringstream istr( paths.c_str() );
    std::string path;
    bool ok = true;
    while (istr >> path)
    {
        ok &= (PathResolver::FindLinkDestClass(context, linktype, path, nullptr) != nullptr);
    }
    return ok;
}




}
