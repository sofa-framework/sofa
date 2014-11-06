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
#ifndef SOFA_HELPER_SYSTEM_FILESYSTEM_H
#define SOFA_HELPER_SYSTEM_FILESYSTEM_H

#include <sofa/SofaFramework.h>

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
/// they provide only basic functionality. In particular, they were only written
/// to work with absolute paths.
///
/// This set of functions is not meant to be complete, but it can be completed
/// if need be.
namespace FileSystem
{

/// @brief List the content of a directory.
///
/// It pushes the filenames (not their absolute paths) in the vector provided in argument.
/// The directory must exist.
/// @return true on error
SOFA_HELPER_API bool listDirectory(const std::string& directoryPath,
                                   std::vector<std::string>& outputFilenames);

/// @brief List the files in a directory which match the given extension.
///
/// It pushes the filenames (not their absolute paths) in the vector provided in argument.
/// The directory must exist.
/// @return true on error
SOFA_HELPER_API bool listDirectory(const std::string& directoryPath,
                                   std::vector<std::string>& outputFilenames,
                                   const std::string& extension);

/// @brief Returns true if and only if the given file exists.
SOFA_HELPER_API bool exists(const std::string& path);

/// @brief Returns true if and only if the given file path corresponds to a directory.
///
/// The path must exist.
SOFA_HELPER_API bool isDirectory(const std::string& path);

};


} // namespace system
} // namespace helper
} // namespace sofa

#endif
