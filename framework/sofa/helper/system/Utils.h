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

#include <sofa/SofaFramework.h>

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

/// @brief Convert a string to a wstring.
///
/// This function uses mbsrtowcs(3) internally, and thus depends on on the
/// LC_CTYPE category of the current locale.
///
/// If you are getting errors from this function, check that your program calls
/// setlocale(LC_CTYPE, "") at the beginning to set the locale according to the
/// environnement variables, and check that those are set to appropriate values.
///
/// @return The converted string on success, or a empty string on failure.
SOFA_HELPER_API std::wstring s2ws(const std::string& s);

/// @brief Convert a wstring to a string.
///
/// This function uses wcstombs(3) internally, and thus depends on
/// on the LC_CTYPE category of the current locale.
///
/// If you are getting errors from this function, check that your program calls
/// setlocale(LC_CTYPE, "") at the beginning to set the locale according to the
/// environnement variables, and check that those are set to appropriate values.
///
/// @return The converted string on success, or a empty string on failure.
SOFA_HELPER_API std::string ws2s(const std::wstring& ws);

#if defined WIN32 || defined _XBOX

/// @brief Simple wrapper around the Windows function GetLastError().
///
/// This function calls ::GetLastError along with the boilerplate code for
/// formatting, and converts the result to a non-wide string with ws2s().
SOFA_HELPER_API std::string GetLastError();

#endif

/// @brief Get the path of the current executable.
///
/// Note that this function uses various non-portable tricks to achieve its
/// goal, and it might not be the most reliable thing ever written.
SOFA_HELPER_API std::string getExecutablePath();

}


} // namespace system
} // namespace helper
} // namespace sofa


#endif
