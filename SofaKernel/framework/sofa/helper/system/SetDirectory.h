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
#ifndef SOFA_HELPER_SYSTEM_SETDIRECTORY_H
#define SOFA_HELPER_SYSTEM_SETDIRECTORY_H

#include <string>

#include <sofa/helper/helper.h>

namespace sofa
{

namespace helper
{

namespace system
{

// A small utility class to temporarly set the current directory to the same as a specified file
class SOFA_HELPER_API SetDirectory
{
public:
    std::string previousDir;
    std::string directory;

    SetDirectory(const char* filename);
    SetDirectory(const std::string& filename);

    ~SetDirectory();

    /// Return true if the given file has an absolute path
    static bool IsAbsolute(const std::string& filename);

    /// Get the current directory
    static std::string GetCurrentDir();

    /// Get the parent directory of a given file, i.e. if given "a/b/c", return "a/b".
    static std::string GetParentDir(const char* filename);

    /// Get the filename from an absolute path description, i.e. if given"a/b/c", return "c"
    static std::string GetFileName(const char* filename);

    /// Get the extension of a file from an absolute path description, i.e. if given"a/b/c.d", return "d"
    static std::string GetExtension(const char* filename);

    /// Get the filename from an absolute path description without extension i.e. if given"a/b/c.d", return "c"
    static std::string GetFileNameWithoutExtension(const char* filename);

    /// Get the full path of the current process. The given filename should be the value of argv[0].
    static std::string GetProcessFullPath(const char* filename);

    /// Get the file relative to a directory, i.e. if given "../e" and "a/b/c", return "a/b/e".
    static std::string GetRelativeFromDir(const char* filename, const char* basename);

    /// Get the file relative to another file path, i.e. if given "../e" and "a/b/c", return "a/e".
    static std::string GetRelativeFromFile(const char* filename, const char* basename);

    /// Get the file relative to current process path, i.e. if given "../e" and "a/b/c", return "a/e".
    static std::string GetRelativeFromProcess(const char* filename, const char* basename=NULL);

};

} // namespace system

} // namespace helper

} // namespace sofa

#endif
