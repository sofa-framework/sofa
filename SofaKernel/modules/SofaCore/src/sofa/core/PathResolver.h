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
#include <sofa/core/config.h>

namespace sofa::core::objectmodel
{
    class Base;
    class BaseData;
    class BaseLink;
    class AbstractDataLink;
    class BaseClass;
}

namespace sofa::core
{

using objectmodel::Base;
using objectmodel::BaseData;
using objectmodel::BaseLink;
using objectmodel::BaseClass;
using objectmodel::AbstractDataLink;

///////////////////////////////////////////////////////////////////////////////////////
/// @brief This class expose an API to query a context to find Base* or a BaseData*.
///////////////////////////////////////////////////////////////////////////////////////
class SOFA_CORE_API PathResolver
{
public:
    static Base* FindBaseFromPath(const Base* base, const std::string& path);
    static BaseData* FindBaseDataFromPath(Base* base, const std::string& path);
    static BaseData* FindBaseDataFromPath(const BaseData* context, const std::string& path);
    static bool PathHasValidSyntax(const std::string& path);

    /////// ALL THE FOLLOWING IS DUPLICATING THE SOFA API.....
    /// The general idea is to move here all code related to path resolution so we can see if there is
    /// duplicated code
    static bool FindDataLinkDest(Base* base, BaseData*& ptr, const std::string& path, const BaseLink* link);

    template<class T>
    static bool CheckPath(Base* base, T*& ptr, const std::string& path, const BaseLink* link)
    {
        void* result = FindLinkDestClass(base, T::GetClass(), path, link);
        return result != nullptr;
    }

    static void* FindLinkDestClass(Base* context, const BaseClass* destType, const std::string& path, const BaseLink* link);

    template<class T>
    static bool CheckPath(Base* context, const std::string& path)
    {
        if (path.empty())
            return false;
        return CheckPath(path, T::GetClass(), context);
    }

    template<class T>
    static bool FindLinkDest(Base* base, T*& ptr, const std::string& path, const BaseLink* link)
    {
        void* result = FindLinkDestClass(base, T::GetClass(), path, link);
        ptr = reinterpret_cast<T*>(result);
        return (result != nullptr);
    }

    static sofa::core::objectmodel::Base* FindLink(sofa::core::objectmodel::Base* base, const std::string& path);

    /// Check that a given path is valid, that the pointed object exists and is of the right type
    static bool CheckPath(sofa::core::objectmodel::Base* context, const std::string& path);
};

}
