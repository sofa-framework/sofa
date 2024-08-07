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
#include <sofa/helper/Utils.h>
#include <sofa/helper/StringUtils.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/FileRepository.h>
#include <algorithm>

#ifdef WIN32
# include <Windows.h>
# include <StrSafe.h>
# include <Shlobj_core.h>
#elif defined __APPLE__
# include <mach-o/dyld.h>       // for _NSGetExecutablePath()
# include <errno.h>
# include <sysdir.h>  // for sysdir_start_search_path_enumeration
# include <glob.h>    // for glob needed to expand ~ to user dir
#else
# include <string.h>            // for strerror()
# include <unistd.h>            // for readlink()
# include <errno.h>
# include <linux/limits.h>      // for PATH_MAX
# include <cstdlib>
# include <sys/types.h>
# include <pwd.h>
#endif

#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <sofa/helper/logging/Messaging.h>


using sofa::helper::system::FileSystem;


namespace sofa::helper
{

std::wstring Utils::widenString(const std::string& s)
{
    return sofa::helper::widenString(s);
}


std::string Utils::narrowString(const std::wstring& ws)
{
    return sofa::helper::narrowString(ws);
}


std::string Utils::downcaseString(const std::string& s)
{
    return sofa::helper::downcaseString(s);
}


std::string Utils::upcaseString(const std::string& s)
{
    return sofa::helper::upcaseString(s);
}


#if defined WIN32
std::string Utils::GetLastError() {
    LPVOID lpErrMsgBuf;
    LPVOID lpMessageBuf;
    const DWORD dwErrorCode = ::GetLastError();

    // Get the string corresponding to the error code
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                  | FORMAT_MESSAGE_FROM_SYSTEM
                  | FORMAT_MESSAGE_IGNORE_INSERTS,
                  nullptr,
                  dwErrorCode,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR)&lpErrMsgBuf,
                  0,
                  nullptr);
    // Allocate a bigger buffer
    lpMessageBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
                                      (lstrlen((LPCTSTR)lpErrMsgBuf)+40)*sizeof(TCHAR));
    // Format the message like so: "error 'code': 'message'"
    StringCchPrintf((LPTSTR)lpMessageBuf,
                    LocalSize(lpMessageBuf) / sizeof(TCHAR),
                    TEXT("error %d: %s"),
                    dwErrorCode, lpErrMsgBuf);

    const std::wstring wsMessage((LPCTSTR)lpMessageBuf);
    LocalFree(lpErrMsgBuf);
    LocalFree(lpMessageBuf);
    return helper::narrowString(wsMessage);
}
#endif

static std::string computeExecutablePath()
{
    std::string path = "";

#if defined(WIN32)
    std::vector<TCHAR> lpFilename(MAX_PATH);
    const int ret = GetModuleFileName(nullptr, /* nullptr --> executable of the current process */
                                      &lpFilename[0],
                                      MAX_PATH);
    if (ret == 0 || ret == MAX_PATH) {
        msg_error("Utils::computeExecutablePath()") << Utils::GetLastError();
    } else {
        path = helper::narrowString(std::wstring(&lpFilename[0]));
    }

#elif defined(__APPLE__)
    std::vector<char> buffer(PATH_MAX);
    std::vector<char> real_path(PATH_MAX);
    uint32_t size = buffer.size();
    if (_NSGetExecutablePath(&buffer[0], &size) != 0) {
        msg_error("Utils::computeExecutablePath()") << "_NSGetExecutablePath() failed";
    }
    if (realpath(&buffer[0], &real_path[0]) == 0) {
        msg_error("Utils::computeExecutablePath()") << "realpath() failed";
    }
    path = std::string(&real_path[0]);

#else  // Linux
    std::vector<char> buffer(PATH_MAX);
    if (readlink("/proc/self/exe", &buffer[0], buffer.size()) == -1) {
        int error = errno;
        msg_error("Utils::computeExecutablePath()") << strerror(error);
    } else {
        path = std::string(&buffer[0]);
    }
#endif

    return FileSystem::cleanPath(path);
}

const std::string& Utils::getExecutablePath()
{
    static const std::string path = computeExecutablePath();
    return path;
}

const std::string& Utils::getExecutableDirectory()
{
    static const std::string path = FileSystem::getParentDirectory(getExecutablePath());
    return path;
}

static std::string computeSofaPathPrefix()
{
    const char* pathVar = getenv("SOFA_ROOT");
    if (pathVar != nullptr && FileSystem::exists(pathVar))
    {
        return FileSystem::convertBackSlashesToSlashes(pathVar);
    }
    else {
        const std::string exePath = Utils::getExecutablePath();
        const std::size_t pos = exePath.rfind("/bin/");
        if (pos == std::string::npos) {
            // This triggers a segfault on MacOS (static call problem): see https://github.com/sofa-framework/sofa/issues/636
            // msg_error("Utils::getSofaPathPrefix()") << "failed to deduce the root path of Sofa from the application path: (" << exePath << ")";

            // Safest thing to return in this case, I guess.
            return Utils::getExecutableDirectory();
        }
        else {
            return exePath.substr(0, pos);
        }
    }
}

const std::string& Utils::getSofaPathPrefix()
{
    static const std::string prefix = computeSofaPathPrefix();
    return prefix;
}

const std::string Utils::getSofaPathTo(const std::string& pathFromBuildDir)
{
    std::string path = FileSystem::append(Utils::getSofaPathPrefix(), pathFromBuildDir);
    if(FileSystem::exists(path))
    {
        return path;
    }
    else
    {
        return Utils::getSofaPathPrefix();
    }
}

std::map<std::string, std::string> Utils::readBasicIniFile(const std::string& path)
{
    std::map<std::string, std::string> map;
    std::ifstream iniFile(path.c_str());
    if (!iniFile.good())
    {
        msg_error("Utils::readBasicIniFile()") << "Error while trying to read file (" << path << ")";
    }

    std::string line;
    while (std::getline(iniFile, line))
    {
        const size_t equalPos = line.find_first_of('=');
        if (equalPos != std::string::npos)
        {
            const std::string key = line.substr(0, equalPos);
            const std::string value = line.substr(equalPos + 1, std::string::npos);
            map[key] = value;
        }
    }

    return map;
}

// no standard/portable way
const std::string& Utils::getUserLocalDirectory()
{

    auto computeUserHomeDirectory = []()
    {
// Windows: "LocalAppData" directory i.e ${HOME}\AppData\Local
#ifdef WIN32
        std::wstring wresult;
        wchar_t* path = nullptr;
        const auto hr = SHGetKnownFolderPath(FOLDERID_LocalAppData, 0, nullptr, &path);
        if (SUCCEEDED(hr))
        {
            wresult = std::wstring(path);
        }
        if (path)
        {
            CoTaskMemFree(path);
        }

        return Utils::narrowString(wresult);

#elif defined(__APPLE__) // macOS : ${HOME}/Library/Application Support
        // https://stackoverflow.com/questions/5123361/finding-library-application-support-from-c
        
        char path[PATH_MAX];
        auto state = sysdir_start_search_path_enumeration(SYSDIR_DIRECTORY_APPLICATION_SUPPORT,
                                                          SYSDIR_DOMAIN_MASK_USER);
        if ((state = sysdir_get_next_search_path_enumeration(state, path)))
        {
            glob_t globbuf;
            if (glob(path, GLOB_TILDE, nullptr, &globbuf) == 0) 
            {
                std::string result(globbuf.gl_pathv[0]);
                globfree(&globbuf);
                return result;
            } 
            else
            {
                // "Unable to expand tilde"
                return std::string("");
            }
        }
        else
        {
            // "Failed to get settings folder"
            return std::string("");
        }
        
#else // Linux: either ${XDG_CONFIG_HOME} if defined, or ${HOME}/.config (should be equivalent)
        const char* configDir;

        // if env.var XDG_CONFIG_HOME is defined
        if ((configDir = std::getenv("XDG_CONFIG_HOME")) == nullptr)
        {
            const char* homeDir;

            // else if HOME is defined
            if ((homeDir = std::getenv("HOME")) == nullptr)
            {
                // else system calls are used
                homeDir = getpwuid(getuid())->pw_dir;
            }

            return std::string(homeDir) + std::string("/.config");
        }
        else
        {
            return std::string(configDir);
        }

#endif
    };

    static std::string homeDir = FileSystem::cleanPath(computeUserHomeDirectory());
    return homeDir;
}

const std::string& Utils::getSofaUserLocalDirectory()
{
    constexpr std::string_view sofaLocalDirSuffix = "SOFA";

    static std::string sofaLocalDirectory = FileSystem::cleanPath(FileSystem::findOrCreateAValidPath(
        FileSystem::append(getUserLocalDirectory(), sofaLocalDirSuffix)));

    return sofaLocalDirectory;
}


} // namespace sofa::helper


