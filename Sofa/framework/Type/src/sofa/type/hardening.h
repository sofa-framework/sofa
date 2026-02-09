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

#include <limits>
#include <type_traits>


// This file should contain useful function to harden (i.e make safer) the code

namespace sofa::type::hardening
{

template<typename IndexType> requires std::is_integral_v<IndexType>
constexpr bool checkOverflow(IndexType a, IndexType b)
{
    if (a <= 0) return false;
    return a > std::numeric_limits<IndexType>::max() / b;
}

inline bool safeStrToInt(const std::string& s, int& result)
{
    char* endptr = nullptr;
    errno = 0;
    long val = std::strtol(s.c_str(), &endptr, 10);
    if (errno != 0 || endptr == s.c_str() || val < std::numeric_limits<int>::min() || val > std::numeric_limits<int>::max())
        return false;
    result = static_cast<int>(val);
    return true;
}

inline bool safeStrToUInt(const std::string& s, unsigned int& result)
{
    char* endptr = nullptr;
    errno = 0;
    unsigned long val = std::strtoul(s.c_str(), &endptr, 10);
    if (errno != 0 || endptr == s.c_str() || val > std::numeric_limits<unsigned int>::max())
        return false;
    result = static_cast<unsigned int>(val);
    return true;
}

template<typename ScalarType> requires std::is_scalar_v<ScalarType>
bool safeStrToScalar(const std::string& s, ScalarType& result)
{
    char* endptr = nullptr;
    errno = 0;
    long double val = std::strtold(s.c_str(), &endptr);
    if (errno != 0 || endptr == s.c_str())
        return false;
    
    result = static_cast<ScalarType>(val);
    return true;
}

/// Escape a string for safe use in a shell double-quoted context.
/// Escapes: backslash, backtick, dollar, double-quote, and newline.
inline std::string escapeForShell(const std::string& input)
{
    std::string result;
    result.reserve(input.size() + 16);
    for (const char c : input)
    {
        switch (c)
        {
        case '\\': result += "\\\\"; break;
        case '"':  result += "\\\""; break;
        case '$':  result += "\\$";  break;
        case '`':  result += "\\`";  break;
        case '\n': result += " ";    break;  // Replace newline with space
        default:   result += c;      break;
        }
    }
    return result;
}

} //namespace sofa::type::hardening
