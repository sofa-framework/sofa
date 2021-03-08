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
#include <sofa/core/config.h>
#include <sofa/core/reflection/fwd.h>
#include <sofa/core/reflection/ClassId.h>
#include <type_traits>
#include <string>

namespace sofa::core::reflection
{

namespace
{
// https://codereview.stackexchange.com/questions/48594/unique-type-id-no-rtti
// Waiting c++20 https://en.cppreference.com/w/cpp/utility/source_location/function_name
template<class T>
static constexpr const char* UID() { 
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__; 
#endif 
}
}

class SOFA_CORE_API Class
{
public:
    /// Returns a ClassId that is unique from each type parameter T.
    template<class T>
    static const sofa::core::reflection::ClassId& GetClassId()
    {
        /// Create the unique ID once, and returns it each time.
        static sofa::core::reflection::ClassId typeId(UID<std::decay_t<T>>());
        return typeId;
    }

    template<class T>
    static const sofa::core::reflection::ClassInfo* GetClassInfo()
    {
        static const sofa::core::reflection::ClassInfo* cachedInfo = GetClassId<T>().getClassInfo();
        return cachedInfo;
    }
};

} /// namespace sofa::defaulttype
