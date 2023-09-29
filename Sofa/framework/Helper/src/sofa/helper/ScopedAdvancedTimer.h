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

#include <optional>
#include <sofa/helper/config.h>
#include<string>

#include <sofa/helper/AdvancedTimer.h>

namespace sofa::helper
{

/// Scoped (RAII) AdvancedTimer to simplify a basic usage
/// Example of use
/// {   ///< open a scope to start measuring
///     ScopedAdvancedTimer t("myMeasurement")
///     ...
///     ...
/// }   ///< close the scope... the timer t is destructed and the
///     measurement recorded.
struct SOFA_HELPER_API ScopedAdvancedTimer
{
    AdvancedTimer::IdStep m_id;
    std::optional<AdvancedTimer::IdObj> m_objId;

    explicit ScopedAdvancedTimer(const std::string& message);
    explicit ScopedAdvancedTimer( const char* message );

    template<class T>
    explicit ScopedAdvancedTimer(const char* message, T* obj);

    ~ScopedAdvancedTimer();
};


template <class T>
ScopedAdvancedTimer::ScopedAdvancedTimer(const char* message, T* obj)
    : m_id(message)
    , m_objId(obj->getName())
{
    AdvancedTimer::stepBegin(m_id, *m_objId);
}

} /// sofa::helper

#ifdef TRACY_ENABLE
    #include <tracy/Tracy.hpp>
    #define SCOPED_TIMER(name) ZoneScopedN(name)
    #define SCOPED_TIMER_VARNAME(varname, name) ZoneNamedN(varname, name, true)
#else
    #define SCOPED_TIMER(name) sofa::helper::ScopedAdvancedTimer timer(name)
    #define SCOPED_TIMER_VARNAME(varname, name) sofa::helper::ScopedAdvancedTimer varname(name)
#endif
