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
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/Utils.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#ifdef WIN32
# include <windows.h>
# include <winerror.h>
# include <strsafe.h>
# include "Shlwapi.h"
# include <shellapi.h>
#else
# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>
# include <errno.h>
# include <string.h>
# include <unistd.h>
#endif

#if defined(__APPLE__)
# include <stdio.h>
# include <spawn.h>
#endif

#ifdef linux
# include <spawn.h>
# include <sys/wait.h>
#endif

#include <cassert>
#include <sofa/helper/system/SetDirectory.h>

namespace fs = std::filesystem;

namespace sofa
{
namespace helper
{
namespace system
{

static bool pathHasDrive(const std::string& path) {
    return path.length() >=3
            && ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z'))
            && path[1] == ':';
}

static std::string pathWithoutDrive(const std::string& path) {
    return path.substr(2, std::string::npos);
}

static std::string pathDrive(const std::string& path) {
    return path.substr(0, 2);
}

// Note: This function uses manual string manipulation instead of std::filesystem::path::parent_path()
// because std::filesystem does not handle trailing slashes the same way.
// For example, std::filesystem treats "/abc/def/ghi/" and "/abc/def/ghi" differently for parent_path(),
// while this implementation normalizes trailing slashes before computing the parent.
// Additionally, Windows drive letters (e.g., "C:/") require special handling to preserve the drive prefix.
static std::string computeParentDirectory(const std::string& path)
{
    if (path == "")
        return ".";
    else if (path == "/")
        return "/";
    else if (path[path.length()-1] == '/')
        return computeParentDirectory(path.substr(0, path.length() - 1));
    else {
        const size_t last_slash = path.find_last_of('/');
        if (last_slash == std::string::npos)
            return ".";
        else if (last_slash == 0)
            return "/";
        else if (last_slash == path.length())
            return "";
        else
            return path.substr(0, last_slash);
    }
}

std::string FileSystem::getExtension(const std::string& filename)
{
    const std::string s = filename;
    const std::string::size_type pos = s.find_last_of('.');
    if (pos == std::string::npos)
        return ""; // no extension
    else
        return s.substr(pos+1);
}

bool FileSystem::listDirectory(const std::string& directoryPath,
                               std::vector<std::string>& outputFilenames)
{
    std::error_code ec;
    fs::directory_iterator it(directoryPath, ec);
    if (ec)
    {
        msg_error("FileSystem::listDirectory()") << directoryPath << ": " << ec.message();
        return true;
    }
    for (; it != fs::directory_iterator(); it.increment(ec))
    {
        if (ec)
        {
            msg_error("FileSystem::listDirectory()") << directoryPath << ": " << ec.message();
            return true;
        }
        const std::string filename = it->path().filename().string();
        if (filename != "." && filename != "..")
            outputFilenames.push_back(filename);
    }
    return false;
}

bool FileSystem::createDirectory(const std::string& path)
{
    std::error_code ec;
    if (fs::exists(path, ec))
    {
        if (!fs::is_directory(path, ec))
        {
            msg_error("FileSystem::createDirectory()") << path << ": File exists and is not a directory";
            return true;
        }
        return false;
    }
    if (ec)
    {
        msg_error("FileSystem::createDirectory()") << path << ": " << ec.message();
        return true;
    }
    fs::create_directories(path, ec);
    if (ec)
    {
        msg_error("FileSystem::createDirectory()") << path << ": " << ec.message();
        return true;
    }
    return false;
}

bool FileSystem::removeDirectory(const std::string& path)
{
#ifdef WIN32
    if (RemoveDirectory(sofa::helper::widenString(path).c_str()) == 0)
    {
        msg_error("FileSystem::removeDirectory()") << path << ": " << Utils::GetLastError();
        return true;
    }
#else
    if (rmdir(path.c_str()))
    {
        msg_error("FileSystem::removeDirectory()") << path << ": " << strerror(errno);
        return true;
    }
#endif
    return false;
}

bool FileSystem::removeAll(const std::string& path){
    std::error_code ec;
    fs::remove_all(path, ec);
    return !ec;
}

bool FileSystem::removeFile(const std::string& path)
{
    std::error_code ec;
    const bool removed = fs::remove(path, ec);
    if (ec)
    {
        msg_error("FileSystem::removeFile()") << path << ": " << ec.message();
        return false;
    }
    return removed;
}

bool FileSystem::exists(const std::string& path, bool quiet)
{
    std::error_code ec;
    const bool result = fs::exists(path, ec);
    if (ec && !quiet)
        msg_error("FileSystem::exists()") << path << ": " << ec.message();
    return result;
}

bool FileSystem::isDirectory(const std::string& path, bool quiet)
{
    std::error_code ec;
    const bool result = fs::is_directory(path, ec);
    if (ec && !quiet)
        msg_error("FileSystem::isDirectory()") << path << ": " << ec.message();
    return result;
}

bool FileSystem::isFile(const std::string &path, bool quiet)
{
    if(!exists(path, quiet))
        return false;

    std::error_code ec;
    const bool result = fs::is_regular_file(path, ec);
    if (ec && !quiet)
        msg_error("FileSystem::isFile()") << path << ": " << ec.message();
    return result;
}

bool FileSystem::isAbsolute(const std::string& path)
{
    return !path.empty()
            && (pathHasDrive(path)
                || path[0] == '/');
}

std::string FileSystem::convertBackSlashesToSlashes(const std::string& path)
{
    std::string str(path);
    std::replace(str.begin(), str.end(), '\\', '/');
    return str;
}

std::string FileSystem::convertSlashesToBackSlashes(const std::string& path)
{
    std::string str(path);
    std::replace(str.begin(), str.end(), '/', '\\');
    return str;
}

std::string FileSystem::removeExtraSlashes(const std::string& path)
{
    std::string str = path;
    size_t pos = str.find("//");
    while(pos != std::string::npos) {
        str.replace(pos, 2, "/");
        pos = str.find("//");
    }

    pos = str.find("/./");
    while(pos != std::string::npos) {
        str.replace(pos, 3, "/");
        pos = str.find("/./");
    }

    return str;
}

std::string FileSystem::removeExtraBackSlashes(const std::string& path)
{
    std::string str = path;
    size_t pos = str.find("\\\\");
    while(pos != std::string::npos) {
        str.replace(pos, 2, "\\");
        pos = str.find("\\\\");
    }

    pos = str.find("\\.\\");
    while(pos != std::string::npos) {
        str.replace(pos, 3, "\\");
        pos = str.find("\\.\\");
    }

    return str;
}

std::string FileSystem::cleanPath(const std::string& path, separator s)
{
    if(s == SLASH)
        return removeExtraSlashes(convertBackSlashesToSlashes(path));
    else
        return removeExtraBackSlashes(convertSlashesToBackSlashes(path));
}

std::string FileSystem::getParentDirectory(const std::string& path)
{
    if (pathHasDrive(path))
        return pathDrive(path) + computeParentDirectory(pathWithoutDrive(path));
    else
        return computeParentDirectory(path);
}

// Note: This function uses manual string manipulation instead of std::filesystem::path::filename()
// because std::filesystem does not handle trailing slashes correctly.
// For example, std::filesystem::path("/abc/def/ghi/").filename() returns "" (empty),
// while this implementation returns "ghi" by recursively stripping trailing slashes first.
// Windows drive letters also require special handling.
std::string FileSystem::stripDirectory(const std::string& path)
{
    if (pathHasDrive(path))
        return stripDirectory(pathWithoutDrive(path));
    else
    {
        const size_t last_slash = path.find_last_of("/");
        if (last_slash == std::string::npos)
            return path;
        else if (last_slash == path.size() - 1)
            if (path.size() == 1)
                return "/";
            else
                return stripDirectory(path.substr(0, path.size() - 1));
        else
            return path.substr(last_slash + 1, std::string::npos);
    }
}

bool FileSystem::listDirectory(const std::string& directoryPath,
                               std::vector<std::string>& outputFilenames,
                               const std::string& extension)
{
    std::error_code ec;
    fs::directory_iterator it(directoryPath, ec);
    if (ec)
    {
        msg_error("FileSystem::listDirectory()") << directoryPath << ": " << ec.message();
        return true;
    }
    for (; it != fs::directory_iterator(); it.increment(ec))
    {
        if (ec)
        {
            msg_error("FileSystem::listDirectory()") << directoryPath << ": " << ec.message();
            return true;
        }
        if (it->is_regular_file(ec))
        {
            const std::string filename = it->path().filename().string();
            if (filename.size() >= extension.size())
            {
                const std::string fileExt = it->path().extension().string();
                if (fileExt == extension || (extension.size() > 0 && fileExt.size() > 0 && fileExt.substr(1) == extension))
                    outputFilenames.push_back(filename);
            }
        }
    }
    return false;
}

int FileSystem::findFiles(const std::string& directoryPath,
                          std::vector<std::string>& outputFilePaths,
                          const std::string& extension, const int depth)
{
    std::error_code ec;
    fs::directory_iterator it(directoryPath, ec);
    if (ec)
    {
        msg_error("FileSystem::findFiles()") << directoryPath << ": " << ec.message();
        return -1;
    }
    for (; it != fs::directory_iterator(); it.increment(ec))
    {
        if (ec)
        {
            msg_error("FileSystem::findFiles()") << directoryPath << ": " << ec.message();
            return -1;
        }
        const std::string filepath = it->path().string();

        if (it->is_directory(ec) && it->path().filename().string()[0] != '.' && depth > 0)
        {
            if (findFiles(filepath, outputFilePaths, extension, depth - 1) == -1)
                return -1;
        }
        else if (it->is_regular_file(ec))
        {
            const std::string filename = it->path().filename().string();
            if (filename.length() >= extension.length() &&
                filename.compare(filename.length() - extension.length(), extension.length(), extension) == 0)
            {
                outputFilePaths.push_back(filepath);
            }
        }
    }
    return static_cast<int>(outputFilePaths.size());
}

void FileSystem::ensureFolderExists(const std::string& pathToFolder)
{
    std::error_code ec;
    if (!fs::exists(pathToFolder, ec))
    {
        if (ec)
        {
            msg_error("FileSystem::ensureFolderExists()") << pathToFolder << ": " << ec.message();
            return;
        }
        fs::create_directories(pathToFolder, ec);
        if (ec)
        {
            msg_error("FileSystem::ensureFolderExists()") << pathToFolder << ": " << ec.message();
        }
    }
}

void FileSystem::ensureFolderForFileExists(const std::string& pathToFile)
{
    fs::path p(pathToFile);
    if (p.has_parent_path())
    {
        std::error_code ec;
        fs::create_directories(p.parent_path(), ec);
        if (ec)
        {
            msg_error("FileSystem::ensureFolderForFileExists()") << pathToFile << ": " << ec.message();
        }
    }
}

std::string FileSystem::append(const std::string_view& existingPath, const std::string_view& toAppend)
{
    if (toAppend.empty())
    {
        return std::string(existingPath);
    }

    constexpr auto isADirectorySeparator = [](const char c) { return c == '/' || c == '\\'; };

    if (isADirectorySeparator(toAppend.front()))
    {
        return append(existingPath, toAppend.substr(1));
    }

    if (!existingPath.empty() && isADirectorySeparator(existingPath.back()))
    {
        return append(existingPath.substr(0, existingPath.size() - 1), toAppend);
    }
    
    return std::string(existingPath) + "/" + std::string(toAppend);
}

bool FileSystem::openFileWithDefaultApplication(const std::string& filename)
{
    bool success = false;

    if (!filename.empty())
    {
        std::error_code ec;
        if (!fs::exists(filename, ec) || ec)
        {
            msg_error("FileSystem::openFileWithDefaultApplication()") << "File does not exist: " << filename;
            return success;
        }

#ifdef WIN32
        if ((INT_PTR)ShellExecuteA(nullptr, "open", filename.c_str(), nullptr, nullptr, SW_SHOWNORMAL) > 32)
            success = true;
#elif defined(__APPLE__)
        pid_t pid;
        char* argv[] = {const_cast<char*>("open"), const_cast<char*>(filename.c_str()), nullptr};
        if (posix_spawn(&pid, "/usr/bin/open", nullptr, nullptr, argv, nullptr) == 0)
        {
            int status;
            if (waitpid(pid, &status, 0) != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0)
                success = true;
        }
#else
        pid_t pid;
        const char* argv[] = {"xdg-open", filename.c_str(), nullptr};
        if (posix_spawn(&pid, "/usr/bin/xdg-open", nullptr, nullptr, const_cast<char* const*>(argv), environ) == 0)
        {
            int status;
            if (waitpid(pid, &status, 0) != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0)
                success = true;
        }
#endif
    }

    return success;
}

} // namespace system
} // namespace helper
} // namespace sofa
