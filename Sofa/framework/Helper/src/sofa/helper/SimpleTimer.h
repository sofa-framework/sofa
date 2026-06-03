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

#include <chrono>
#include <string>

#include <sofa/helper/logging/Messaging.h>

namespace sofa::helper
{
template<class Unit>
struct UnitInfo;

template<> struct UnitInfo<std::chrono::nanoseconds>
{
    static constexpr std::string_view unit = "ns";
};

template<> struct UnitInfo<std::chrono::microseconds>
{
    static constexpr std::string_view unit = "us";
};

template<> struct UnitInfo<std::chrono::milliseconds>
{
    static constexpr std::string_view unit = "ms";
};

template<> struct UnitInfo<std::chrono::seconds>
{
    static constexpr std::string_view unit = "s";
};

/**
 * @class SimpleTimer
 * @brief A RAII utility class for measuring elapsed time in operations and log it.
 *
 * @code
 * // Measure a block of code
 * {
 *     sofa::helper::SimpleTimer<std::chrono::milliseconds> timer("File I/O Simulation");
 *     // ... Simulate file reading/writing here ...
 *     // The elapsed time is logged by the timer at the end of the scope
 * }
 * @endcode
 */
template<class Unit = std::chrono::nanoseconds>
struct SimpleTimer
{
    std::string m_name;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    bool hasEnded { false };

    explicit SimpleTimer(const std::string& name)
        : m_name(name)
        , m_start(std::chrono::high_resolution_clock::now())
    {}

    ~SimpleTimer()
    {
        stop();
    }

    void restart()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        if (!hasEnded)
        {
            auto end = std::chrono::high_resolution_clock::now();
            msg_info("Timer") << m_name << " took " << std::chrono::duration_cast<Unit>(end - m_start).count() << UnitInfo<Unit>::unit;
            hasEnded = true;
        }
    }


};

}

