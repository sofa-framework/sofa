/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_HELPER_SYSTEM_UTILS_H
#define SOFA_HELPER_SYSTEM_UTILS_H

#include <sofa/helper/helper.h>

#include <string>

namespace sofa
{
namespace helper
{
namespace system
{

/// @brief Contains possibly useful functions, somehow related to system
/// aspects, that don't fit anywhere else.
namespace Utils
{

#ifdef WIN32

/// @brief Convert a string to a wstring.
///
/// This will purely and simply truncate 'wchar' to 'char', which is terrible,
/// but should work at least for ASCII characters. This is useful under Windows,
/// because Sofa is compiled with Unicode support, and thus functions from the
/// Windows API use wchar (e.g. for error messages), which can be a pain in some
/// cases.
SOFA_HELPER_API std::wstring s2ws(const std::string& s);

/// @brief Convert a wstring to a string.
SOFA_HELPER_API std::string ws2s(const std::wstring& ws);

/// @brief Simple wrapper around the Windows function GetLastError().
///
/// This function calls ::GetLastError along with the boilerplate code for
/// formatting, and converts the result to a non-wide string with ws2s().
SOFA_HELPER_API std::string GetLastError();

#endif

}


} // namespace system
} // namespace helper
} // namespace sofa


#endif
