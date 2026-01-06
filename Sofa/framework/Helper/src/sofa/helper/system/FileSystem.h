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
#ifndef SOFA_HELPER_SYSTEM_FILESYSTEM_H
#define SOFA_HELPER_SYSTEM_FILESYSTEM_H

#include <sofa/helper/config.h>

#include <stdexcept>
#include <string>
#include <vector>


namespace sofa::helper::system
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

typedef enum { SLASH, BACKSLASH } separator;

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

/// @brief Find the files from a directory which match the given extension.
///
/// It pushes the files absolute paths in the vector provided in argument.
/// @warning The directory must exist.
/// @return the number of files found or -1 as error
static int findFiles(const std::string& directoryPath,
                          std::vector<std::string>& outputFilePaths,
                          const std::string& extension, const int depth = 0);

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

/// @brief The file or empty directory identified by the path is deleted
///
/// @return true if the file was deleted, false if it did not exist.
static bool removeFile(const std::string& path);

/// @brief check that all element in the path exists or create them. (This function accepts relative paths)
///
/// @return the valid path.
SOFA_HELPER_FILESYSTEM_FINDORCREATEAVALIDPATH_DISABLED()
static std::string findOrCreateAValidPath(const std::string path) = delete;

/// @brief Ensures that a folder exists at the specified path. If the folder does not exist, it will be created.
///
/// @param pathToFolder The path of the folder to ensure exists.
///
/// @note The function assumes that a path to a folder is given.
static void ensureFolderExists(const std::string& pathToFolder);

/// @brief Ensures that the folder containing the specified file path exists. If the folder does not
/// exist, it will be created. If the file does not exist, it will not be created.
///
/// @param pathToFile The path to the file. The function ensures that the directory containing this file exists.
///
/// @note The function assumes that a path to a file is given.
static void ensureFolderForFileExists(const std::string& pathToFile);

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

/// @brief Replace slashes with backslashes.
static std::string convertSlashesToBackSlashes(const std::string& path);

/// @brief Replace consecutive occurrences of '/' with a single '/'.
static std::string removeExtraSlashes(const std::string& path);

/// @brief Replace consecutive occurrences of '\' with a single '\'.
static std::string removeExtraBackSlashes(const std::string& path);

/// @brief Clean path (backslashes, extra slashes...)
static std::string cleanPath(const std::string& path, separator s = SLASH);

/// @brief Strip the last component from a path.
/// Return the path given in argument with its last non-slash
/// component and trailing slashes removed, or "." if the path
/// contains no slashes.
/// E.g. /a/b/c --> /a/b
static std::string getParentDirectory(const std::string& path);

/// @brief Strip the directory components from a path.
/// E.g. /a/b/c --> c
static std::string stripDirectory(const std::string& path);


/// Appends a string to an existing path, ensuring exactly one directory
/// separator (/) between each part.
static std::string append(const std::string_view& existingPath, const std::string_view& toAppend);

/// Appends strings to an existing path, ensuring exactly one directory
/// separator (/) between each part.
template<typename... Args>
static std::string append(const std::string_view& existingPath, const std::string_view& toAppend, Args&&... args)
{
    return append(append(existingPath, toAppend), args...);
}

};


} // namespace sofa::helper::system


#endif
