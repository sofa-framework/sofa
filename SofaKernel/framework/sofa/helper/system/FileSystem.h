/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_HELPER_SYSTEM_FILESYSTEM_H
#define SOFA_HELPER_SYSTEM_FILESYSTEM_H

#include <sofa/helper/helper.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace sofa
{
namespace helper
{
namespace system
{

/// @brief Contains functions to interact with the file system.
///
/// Those functions are here only to avoid depending on an external library, and
/// they provide only basic functionality. In particular, most of them were only
/// written to work with absolute paths.
///
/// This set of functions is not meant to be complete, but it can be completed
/// if need be.
class SOFA_HELPER_API FileSystem
{
public:

/// @brief Get the extension of a file from an absolute path description
/// @return i.e. if given"a/b/c.d", return "d"
static std::string getExtension(const std::string& filename);

/// @brief List the content of a directory.
///
/// It pushes the filenames (not their absolute paths) in the vector provided in argument.
/// @warning The directory must exist.
/// @return true on error
static bool listDirectory(const std::string& directoryPath,
                          std::vector<std::string>& outputFilenames);

/// @brief List the files in a directory which match the given extension.
///
/// It pushes the filenames (not their absolute paths) in the vector provided in argument.
/// @warning The directory must exist.
/// @return true on error
static bool listDirectory(const std::string& directoryPath,
                          std::vector<std::string>& outputFilenames,
                          const std::string& extension);

/// @brief Create a directory. (This function accepts relative paths)
///
/// On Unix platform, the directory is created with permissions 0755. On
/// Windows, it is created with no special attributes.
/// @return true on error
static bool createDirectory(const std::string& path);

/// @brief Remove an empty directory. (This function accepts relative paths)
///
/// @return true on error
static bool removeDirectory(const std::string& path);

/// @brief Remove a non-empty directory. (This function accepts relative paths)
///
/// @return true on error
static bool removeAll(const std::string& path) ;

/// @brief check that all element in the path exists or create them. (This function accepts relative paths)
///
/// @return the valid path.
static std::string findOrCreateAValidPath(const std::string path) ;

/// @brief Return true if and only if the given file exists.
static bool exists(const std::string& path);

/// @brief Return true if and only if the given file path corresponds to a directory.
///
/// @warning The path must exist.
static bool isDirectory(const std::string& path);

/// @brief Return true if and only if the given file path is absolute.
static bool isAbsolute(const std::string& path);

/// @brief Return true if and only if the given file path is an existing file.
static bool isFile(const std::string& path);

/// @brief Replace backslashes with slashes.
static std::string convertBackSlashesToSlashes(const std::string& path);

/// @brief Replace consecutive occurrences of '/' with a single '/'.
static std::string removeExtraSlashes(const std::string& path);

/// @brief Clean path (backslashes, extra slashes...)
static std::string cleanPath(const std::string& path);

/// @brief Strip the last component from a path.
/// Return the path given in argument with its last non-slash
/// component and trailing slashes removed, or "." if the path
/// contains no slashes.
/// E.g. /a/b/c --> /a/b
static std::string getParentDirectory(const std::string& path);

/// @brief Strip the directory components from a path.
/// E.g. /a/b/c --> c
static std::string stripDirectory(const std::string& path);

};


} // namespace system
} // namespace helper
} // namespace sofa

#endif
