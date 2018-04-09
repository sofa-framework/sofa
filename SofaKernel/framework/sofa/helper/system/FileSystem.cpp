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
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/logging/Messaging.h>
#include <sofa/helper/Utils.h>

#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>
#ifdef WIN32
# include <windows.h>
# include <winerror.h>
# include <strsafe.h>
# include "Shlwapi.h"           // for PathFileExists()
#elif defined(_XBOX)
# include <xtl.h>
#else
# include <dirent.h>
# include <sys/stat.h>
# include <sys/types.h>
# include <errno.h>
# include <string.h>            // for strerror()
# include <unistd.h>
#endif

#include <cassert>
#include "SetDirectory.h"

namespace sofa
{
namespace helper
{
namespace system
{

std::string FileSystem::getExtension(const std::string& filename)
{
    std::string s = filename;
    std::string::size_type pos = s.find_last_of('.');
    if (pos == std::string::npos)
        return ""; // no extension
    else
        return s.substr(pos+1);
}


#if defined(WIN32)
// Helper: call FindFirstFile, taking care of wstring to string conversion.
static HANDLE helper_FindFirstFile(std::string path, WIN32_FIND_DATA *ffd)
{
    TCHAR szDir[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.
    StringCchCopy(szDir, MAX_PATH, Utils::widenString(path).c_str());
    StringCchCat(szDir, MAX_PATH, TEXT("\\*"));

    // Find the first file in the directory.
    hFind = FindFirstFile(szDir, ffd);

    return hFind;
}
#elif defined (_XBOX)
static HANDLE helper_FindFirstFile(std::string path, WIN32_FIND_DATA *ffd)
{
    char szDir[MAX_PATH];
    HANDLE hFind = INVALID_HANDLE_VALUE;

    // Prepare string for use with FindFile functions.  First, copy the
    // string to a buffer, then append '\*' to the directory name.
    strcpy_s(szDir, MAX_PATH, path.c_str());
    strcat_s(szDir, MAX_PATH, "\\*");

    // Find the first file in the directory.
    hFind = FindFirstFile(szDir, ffd);

    return hFind;
}
#endif


bool FileSystem::listDirectory(const std::string& directoryPath,
                               std::vector<std::string>& outputFilenames)
{
#if defined(WIN32) || defined (_XBOX)
    // Find the first file in the directory.
    WIN32_FIND_DATA ffd;
    HANDLE hFind = helper_FindFirstFile(directoryPath, &ffd);
    if (hFind == INVALID_HANDLE_VALUE) {
        msg_error("FileSystem::listDirectory()") << directoryPath << ": " << Utils::GetLastError();
        return false;
    }

    // Iterate over files and push them in the output vector
    do {
# if defined (_XBOX)
        std::string filename = ffd.cFileName;
# else
        std::string filename = Utils::narrowString(ffd.cFileName);
# endif
        if (filename != "." && filename != "..")
            outputFilenames.push_back(filename);
    } while (FindNextFile(hFind, &ffd) != 0);

    // Check for errors
    bool errorOccured = ::GetLastError() != ERROR_NO_MORE_FILES;
    if (errorOccured)
        msg_error("FileSystem::listDirectory()") << directoryPath << ": " << Utils::GetLastError();

    FindClose(hFind);
    return errorOccured;
#else
    DIR *dp = opendir(directoryPath.c_str());
    if (dp == NULL) {
        msg_error("FileSystem::listDirectory()") << directoryPath << ": " << strerror(errno);
        return true;
    }

    struct dirent *ep;
    while ( (ep = readdir(dp)) ) {
        const std::string filename(ep->d_name);
        if (filename != "." && filename != "..")
            outputFilenames.push_back(std::string(ep->d_name));
    }
    closedir(dp);
    return false;
#endif
}

bool FileSystem::createDirectory(const std::string& path)
{
    std::string error;
#ifdef WIN32
    if (CreateDirectory(Utils::widenString(path).c_str(), NULL) == 0)
    {
        DWORD errorCode = ::GetLastError();
        msg_error("FileSystem::createdirectory()") << path << ": " << Utils::GetLastError();
        return true;
    }
#else
    if (mkdir(path.c_str(), 0755))
    {
        msg_error("FileSystem::createdirectory()") << path << ": " << strerror(errno);
        return true;
    }
#endif
    else
    {
        return false;
    }
}


bool FileSystem::removeDirectory(const std::string& path)
{
#ifdef WIN32
    if (RemoveDirectory(Utils::widenString(path).c_str()) == 0)
    {
        DWORD errorCode = ::GetLastError();
        msg_error("FileSystem::removedirectory()") << path << ": " << Utils::GetLastError();
        return true;
    }
#else
    if (rmdir(path.c_str()))
    {
        msg_error("FileSystem::removedirectory()") << path << ": " << strerror(errno);
        return true;
    }
#endif
    return false;
}


bool FileSystem::exists(const std::string& path)
{
#if defined(WIN32)
    ::SetLastError(0);
    if (PathFileExists(Utils::widenString(path).c_str()) != 0)
        return true;
    else
    {
        DWORD errorCode = ::GetLastError();
        if (errorCode != ERROR_FILE_NOT_FOUND && errorCode != ERROR_PATH_NOT_FOUND) // not No such file error
            msg_error("FileSystem::exists()") << path << ": " << Utils::GetLastError();
        return false;
    }

#elif defined (_XBOX)
    DWORD fileAttrib = GetFileAttributes(path.c_str());
    return fileAttrib != -1;
#else
    struct stat st_buf;
    if (stat(path.c_str(), &st_buf) == 0)
        return true;
    else
        if (errno == ENOENT)    // No such file or directory
            return false;
        else {
            msg_error("FileSystem::exists()") << path << ": " << strerror(errno);
            return false;
        }
#endif
}


bool FileSystem::isDirectory(const std::string& path)
{
#if defined(WIN32)
    DWORD fileAttrib = GetFileAttributes(Utils::widenString(path).c_str());
    if (fileAttrib == INVALID_FILE_ATTRIBUTES) {
        msg_error("FileSystem::isDirectory()") << path << ": " << Utils::GetLastError();
        return false;
    }
    else
        return (fileAttrib & FILE_ATTRIBUTE_DIRECTORY) != 0;
#elif defined (_XBOX)
    DWORD fileAttrib = GetFileAttributes(path.c_str());
    if (fileAttrib == -1) {
        msg_error("FileSystem::isDirectory()") << path << ": " << Utils::GetLastError();
        return false;
    }
    else
        return (fileAttrib & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat st_buf;
    if (stat(path.c_str(), &st_buf) != 0) {
        msg_error("FileSystem::isDirectory()") << path << ": " << strerror(errno);
        return false;
    }
    else
        return S_ISDIR(st_buf.st_mode);
#endif
}

bool FileSystem::listDirectory(const std::string& directoryPath,
                               std::vector<std::string>& outputFilenames,
                               const std::string& extension)
{
    // List directory
    std::vector<std::string> files;
    if (listDirectory(directoryPath, files))
        return true;

    // Filter files
    for (std::size_t i=0 ; i!=files.size() ; i++) {
        const std::string& filename = files[i];
        if (filename.size() >= extension.size())
            if (filename.compare(filename.size()-extension.size(),
                                 std::string::npos, extension) == 0)
                outputFilenames.push_back(filename);
    }
    return false;
}


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

bool FileSystem::isAbsolute(const std::string& path)
{
    return !path.empty()
            && (pathHasDrive(path)
                || path[0] == '/');
}

bool FileSystem::isFile(const std::string &path)
{
    return
            FileSystem::exists(path) &&
            !FileSystem::isDirectory(path)
    ;
}

std::string FileSystem::convertBackSlashesToSlashes(const std::string& path)
{
    std::string str = path;
    size_t backSlashPos = str.find('\\');
    while(backSlashPos != std::string::npos)
    {
        str[backSlashPos] = '/';
        backSlashPos = str.find("\\");
    }
    return str;
}

bool FileSystem::removeAll(const std::string& path){
    try{
        boost::filesystem::remove_all(path) ;
    }catch(boost::filesystem::filesystem_error const & e){ return false ; }
    return true ;
}

std::string FileSystem::removeExtraSlashes(const std::string& path)
{
    std::string str = path;
    size_t pos = str.find("//");
    while(pos != std::string::npos) {
        str.replace(pos, 2, "/");
        pos = str.find("//");
    }
    return str;
}


std::string FileSystem::findOrCreateAValidPath(const std::string path)
{
    if( FileSystem::exists(path) )
        return path ;

    std::string parentPath = FileSystem::getParentDirectory(path) ;
    std::string currentFile = FileSystem::stripDirectory(path) ;
    FileSystem::createDirectory(findOrCreateAValidPath( parentPath )+"/"+currentFile) ;
    return path ;
}



std::string FileSystem::cleanPath(const std::string& path)
{
    return removeExtraSlashes(convertBackSlashesToSlashes(path));
}

static std::string computeParentDirectory(const std::string& path)
{
    if (path == "")
        return ".";
    else if (path == "/")
        return "/";
    else if (path[path.length()-1] == '/')
        return computeParentDirectory(path.substr(0, path.length() - 1));
    else {
        size_t last_slash = path.find_last_of('/');
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

std::string FileSystem::getParentDirectory(const std::string& path)
{
    if (pathHasDrive(path))     // check for Windows drive
        return pathDrive(path) + computeParentDirectory(pathWithoutDrive(path));
    else
        return computeParentDirectory(path);
}

std::string FileSystem::stripDirectory(const std::string& path)
{
    if (pathHasDrive(path))     // check for Windows drive
        return stripDirectory(pathWithoutDrive(path));
    else
    {
        size_t last_slash = path.find_last_of("/");
        if (last_slash == std::string::npos)    // No slash
            return path;
        else if (last_slash == path.size() - 1) // Trailing slash
            if (path.size() == 1)
                return "/";
            else
                return stripDirectory(path.substr(0, path.size() - 1));
        else
            return path.substr(last_slash + 1, std::string::npos);
        return "";
    }
}


} // namespace system
} // namespace helper
} // namespace sofa
