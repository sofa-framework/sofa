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
#ifndef SOFA_HELPER_UTILS_H
#define SOFA_HELPER_UTILS_H

#include <sofa/helper/config.h>

#include <string>
#include <map>


namespace sofa::helper
{

/// @brief Contains possibly useful functions, that don't fit anywhere else.
class SOFA_HELPER_API Utils
{
public:

/// @brief Convert a string to a wstring.
///
/// @return The converted string on success, or a empty string on failure.
SOFA_HELPER_UTILS_IN_STRINGUTILS_DEPRECATED()
static std::wstring widenString(const std::string& s);

/// @brief Convert a wstring to a string.
///
/// @return The converted string on success, or a empty string on failure.
SOFA_HELPER_UTILS_IN_STRINGUTILS_DEPRECATED()
static std::string narrowString(const std::wstring& ws);

/// @brief Convert a string to lower case.
SOFA_HELPER_UTILS_IN_STRINGUTILS_DEPRECATED()
static std::string downcaseString(const std::string& s);

/// @brief Convert a string to upper case.
SOFA_HELPER_UTILS_IN_STRINGUTILS_DEPRECATED()
static std::string upcaseString(const std::string& s);

#if defined WIN32

/// @brief Simple wrapper around the Windows function GetLastError().
///
/// This function calls ::GetLastError along with the boilerplate code for
/// formatting, and converts the result to a non-wide string with narrowString().
static std::string GetLastError();

#endif

/// @brief Get the path of the executable that is currently running.
///
/// Note that this function uses various non-portable tricks to achieve its
/// goal, and it might not be the most reliable thing ever written.
static const std::string& getExecutablePath();

/// @brief Get the path to the directory of the executable that is currently running.
static const std::string& getExecutableDirectory();

/// @brief Get the path to the current user local config directory.
static const std::string& getUserLocalDirectory();

/// @brief Get the path to the SOFA directory into the current user local config directory.
static const std::string& getSofaUserLocalDirectory();


/// @brief Get the path to the "root" path of Sofa (i.e. the build directory or
/// the installation prefix).
///
/// @warning This function is meant to be used only by the applications that are
/// distributed with SOFA
/// @return The ABSOLUTE path of Sofa build dir (or install dir)
static const std::string& getSofaPathPrefix();

/// @brief Construct a path based on the build dir path of Sofa
///
/// @warning This function is meant to be used only by the applications that are
/// distributed with SOFA: it uses getSofaPathPrefix()
/// @return The ABSOLUTE path of anything in Sofa build dir (or install dir)
static const std::string getSofaPathTo(const std::string& pathFromBuildDir);

/// @brief Read a file written in a very basic ini-like format.
///
/// For each line that contains a '=' character, (e.g. "key=value"), the returned
/// map will contains a pair <"key", "value">.  Other lines will be ignored.
static std::map<std::string, std::string> readBasicIniFile(const std::string& path);

};


} // namespace sofa::helper


#endif
