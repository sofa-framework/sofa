/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileSystem.h>
#ifdef WIN32
#include <windows.h>
#include <direct.h>
#elif defined(_XBOX)
#include <xtl.h>
#else
#include <unistd.h>
#endif
#if defined (__APPLE__)
#include <sys/param.h>
#include <mach-o/dyld.h>
#endif
#include <string.h>
#include <iostream>

#include <sofa/helper/logging/Messaging.h>

namespace sofa
{

namespace helper
{

namespace system
{

#if defined(WIN32)
    #define chdir _chdir
    #define getcwd _getcwd
#elif defined(_XBOX)
    int chdir(const char* path) { return -1; } // NOT IMPLEMENTED
    char* getcwd(char *buffer, int maxlen) { return ""; } // NOT IMPLEMENTED
#elif defined(PS3)
    std::string g_currentWorkingDir = std::string("/app_home/");
    char* getcwd(char *buffer, int maxlen) { strcpy(buffer, g_currentWorkingDir.c_str()); return buffer;}
    int chdir(const char* path) { g_currentWorkingDir = path; return 1;}
#endif

SetDirectory::SetDirectory(const char* filename)
{
    directory = GetParentDir(filename);
    if (!directory.empty())
    {
        previousDir = GetCurrentDir();
        if (chdir(directory.c_str()) != 0)
            msg_error("SetDirectory") << "can't change directory.";
    }
}

SetDirectory::SetDirectory(const std::string& filename)
{
    SetDirectory(filename.c_str());
}

SetDirectory::~SetDirectory()
{
    if (!directory.empty() && !previousDir.empty())
    {
        if (chdir(previousDir.c_str()) != 0)
            msg_error("SetDirectory") << "can't change directory.";
    }
}

/// Return true if the given file has an absolute path
bool SetDirectory::IsAbsolute(const std::string& filename)
{
    if (filename.empty()) return false;
    if (filename[0] == '/' || filename[0] == '\\') return true;
#ifdef WIN32
    if (filename.length() >= 2 && ((filename[0]>='a' && filename[0]<='z') || (filename[0]>='A' && filename[0]<='Z')) && filename[1]==':') return true;
#endif
    return false;
}

/// Get the current directory
std::string SetDirectory::GetCurrentDir()
{
    char dir[1024];
    memset(dir,0,sizeof(dir));
    if (getcwd(dir, sizeof(dir)) == NULL)
        msg_error("SetDirectory") << "can't get current directory.";
    return dir;
}

std::string SetDirectory::GetParentDir(const char* filename)
{
    std::string s = filename;
    std::string::size_type pos = s.find_last_of("/\\");
    if (pos == std::string::npos)
        return ""; // no directory
    else
        return s.substr(0,pos);
}

std::string SetDirectory::GetFileName(const char* filename)
{
    std::string s = filename;
    std::string::size_type pos = s.find_last_of("/\\");
    if (pos == std::string::npos)
        return s; // no directory
    else
        return s.substr(pos+1);
}

std::string SetDirectory::GetFileNameWithoutExtension(const char* filename)
{
    std::string s = GetFileName(filename);
    std::string::size_type pos = s.find_first_of(".");
    if (pos == std::string::npos)
        return s; // no directory
    else
        return s.substr(0,pos);
}

std::string SetDirectory::GetExtension(const char* filename)
{
    return FileSystem::getExtension(filename) ;
}

std::string SetDirectory::GetRelativeFromDir(const char* filename, const char* basename)
{
    if (!filename || !filename[0]) return "";
    if (IsAbsolute(filename)) return filename; // absolute path
    std::string base = basename;
    std::string s = filename;
    // remove any ".."
    while ((s.substr(0,3)=="../" || s.substr(0,3)=="..\\") && !base.empty())
    {
        s = s.substr(3);
        base = GetParentDir(base.c_str());
    }
    if (base.empty())
        return s;
    else if (base[base.length()-1] == '/')
        return base + s;
    else
        return base + "/" + s;
}

std::string SetDirectory::GetRelativeFromFile(const char* filename, const char* basename)
{
    std::string base = GetParentDir(basename);
    return GetRelativeFromDir(filename, base.c_str());
}

std::string SetDirectory::GetRelativeFromProcess(const char* filename, const char* basename)
{
    std::string base = GetProcessFullPath(basename);
    return GetRelativeFromFile(filename, base.c_str());
}

/// Get the full path of the current process. The given filename should be the value of argv[0].
std::string SetDirectory::GetProcessFullPath(const char* filename)
{
#if defined (WIN32)
    if (!filename || !filename[0])
    {
//       //return __argv[0];
//       int n=0;
        //LPWSTR wpath = *CommandLineToArgvW(GetCommandLineW(),&n);
        //if (wpath)
        //{
        //    char path[1024];
        //    memset(path,0,sizeof(path));
        //    wcstombs(path, wpath, sizeof(path)-1);
        //    if (path[0]) return path;
        //   }
        TCHAR tpath[1024];
        GetModuleFileName(NULL,tpath,1024);
        std::wstring wprocessPath = tpath;
        std::string processPath;
        processPath.assign(wprocessPath.begin(), wprocessPath.end() );
        return processPath;
    }
    /// \TODO use GetCommandLineW and/or CommandLineToArgvW. This is however not strictly necessary, as argv[0] already contains the full path in most cases.
#elif defined (__linux__)
    if (!filename || filename[0]!='/')
    {
        char path[1024];
        memset(path,0,sizeof(path));
        if (readlink("/proc/self/exe",path,sizeof(path)-1) == -1)
            msg_error("SetDirectory") << "can't read the contents of the link.";
        if (path[0])
            return path;
        else
            msg_error("SetDirectory") << "can't get current process path..." ;
    }
#elif defined (__APPLE__)
    if (!filename || filename[0]!='/')
    {
        char* path = new char[4096];
        uint32_t size;
        if ( _NSGetExecutablePath( path, &size ) != 0)
        {
            //realloc
            delete [] path;
            path = new char[size];
            _NSGetExecutablePath( path, &size );
        }
        std::string finalPath(path);
        delete [] path;
        return finalPath;
    }
#endif

    if(filename)
        return filename;
    else return std::string("");
}

} // namespace system

} // namespace helper

} // namespace sofa

