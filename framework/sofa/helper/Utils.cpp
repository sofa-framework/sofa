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
#include <sofa/helper/Utils.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Logger.h>

#ifdef WIN32
# include <Windows.h>
# include <StrSafe.h>
#elif defined _XBOX
# include <xtl.h>
#elif defined __APPLE__
# include <mach-o/dyld.h>       // for _NSGetExecutablePath()
# include <errno.h>
#else
# include <unistd.h>            // for readlink()
# include <errno.h>
# include <linux/limits.h>      // for PATH_MAX
#endif

#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>


using sofa::helper::system::FileSystem;

namespace sofa
{
namespace helper
{

std::wstring Utils::widenString(const std::string& s)
{
    const char * src = s.c_str();
    // Call mbsrtowcs() once to find out the length of the converted string.
    size_t length = mbsrtowcs(NULL, &src, 0, NULL);
    if (length == size_t(-1)) {
        int error = errno;
        Logger::getMainLogger().log(Logger::Warning, strerror(error), "Utils::widenString()");
        return L"";
    }

    // Call mbsrtowcs() again with a correctly sized buffer to actually do the conversion.
    wchar_t * buffer = new wchar_t[length + 1];
    length = mbsrtowcs(buffer, &src, length + 1, NULL);
    if (length == size_t(-1)) {
        int error = errno;
        Logger::getMainLogger().log(Logger::Warning, strerror(error), "Utils::widenString()");
        delete[] buffer;
        return L"";
    }

    if (src != NULL) {
        Logger::getMainLogger().log(Logger::Warning, "Conversion failed (\"" + s + "\")", "Utils::widenString()");
        delete[] buffer;
        return L"";
    }

    std::wstring result(buffer);
    delete[] buffer;
    return result;
}


std::string Utils::narrowString(const std::wstring& ws)
{
    const wchar_t * src = ws.c_str();
    // Call wcstombs() once to find out the length of the converted string.
    size_t length = wcstombs(NULL, src, 0);
    if (length == size_t(-1)) {
        Logger::getMainLogger().log(Logger::Warning, "Conversion failed", "Utils::narrowString()");
        return "";
    }

    // Call wcstombs() again with a correctly sized buffer to actually do the conversion.
    char * buffer = new char[length + 1];
    length = wcstombs(buffer, src, length + 1);
    if (length == size_t(-1)) {
        Logger::getMainLogger().log(Logger::Warning, "Conversion failed", "Utils::narrowString()");
        delete[] buffer;
        return "";
    }

    std::string result(buffer);
    delete[] buffer;
    return result;
}

#if defined WIN32 || defined _XBOX
# ifdef WIN32
std::string Utils::GetLastError() {
    LPVOID lpErrMsgBuf;
    LPVOID lpMessageBuf;
    DWORD dwErrorCode = ::GetLastError();

    // Get the string corresponding to the error code
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                  | FORMAT_MESSAGE_FROM_SYSTEM
                  | FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL,
                  dwErrorCode,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR)&lpErrMsgBuf,
                  0,
                  NULL);
    // Allocate a bigger buffer
    lpMessageBuf = (LPVOID)LocalAlloc(LMEM_ZEROINIT,
                                      (lstrlen((LPCTSTR)lpErrMsgBuf)+40)*sizeof(TCHAR));
    // Format the message like so: "error 'code': 'message'"
    StringCchPrintf((LPTSTR)lpMessageBuf,
                    LocalSize(lpMessageBuf) / sizeof(TCHAR),
                    TEXT("error %d: %s"),
                    dwErrorCode, lpErrMsgBuf);

    std::wstring wsMessage((LPCTSTR)lpMessageBuf);
    LocalFree(lpErrMsgBuf);
    LocalFree(lpMessageBuf);
    return narrowString(wsMessage);
}
# else  // XBOX
std::string Utils::GetLastError() {
    DWORD dwErrorCode = ::GetLastError();
    char buffer[32];
    sprintf_s(buffer, 32, "0x%08.8X", dwErrorCode);
    return buffer;
}
# endif
#endif

static std::string computeExecutablePath()
{
    std::string path = "";

#if defined(_XBOX) || defined(PS3)
    Logger::getMainLogger().log(Logger::Error, "Utils::getExecutablePath() is not implemented on this platform.");

#elif defined(WIN32)
    std::vector<TCHAR> lpFilename(MAX_PATH);
    int ret = GetModuleFileName(NULL, /* NULL --> executable of the current process */
        &lpFilename[0],
        MAX_PATH);
    if (ret == 0 || ret == MAX_PATH) {
        Logger::getMainLogger().log(Logger::Error, Utils::GetLastError(), "Utils::getExecutablePath()");
    } else {
        path = Utils::narrowString(std::wstring(&lpFilename[0]));
    }

#elif defined(__APPLE__)
    std::vector<char> buffer(PATH_MAX);
    std::vector<char> real_path(PATH_MAX);
    uint32_t size = buffer.size();
    if (_NSGetExecutablePath(&buffer[0], &size) != 0) {
        Logger::getMainLogger().log(Logger::Error, "_NSGetExecutablePath() failed", "Utils::getExecutablePath()");
    }
    if (realpath(&buffer[0], &real_path[0]) == 0) {
        Logger::getMainLogger().log(Logger::Error, "realpath() failed", "Utils::getExecutablePath()");
    }
    path = std::string(&real_path[0]);

#else  // Linux
    std::vector<char> buffer(PATH_MAX);
    if (readlink("/proc/self/exe", &buffer[0], buffer.size()) == -1) {
        int error = errno;
        Logger::getMainLogger().log(Logger::Error, strerror(error), "Utils::getExecutablePath()");
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
    const std::string exePath = Utils::getExecutablePath();
    std::size_t pos = exePath.rfind("/bin/");
    if (pos == std::string::npos) {
        Logger::getMainLogger().log(Logger::Error, "failed to deduce the root path of Sofa from the application path: (" + exePath + ")", "Utils::getSofaPathPrefix()");
        // Safest thing to return in this case, I guess.
        return Utils::getExecutableDirectory();
    }
    else {
        return exePath.substr(0, pos);
    }
}

const std::string& Utils::getSofaPathPrefix()
{
    static const std::string prefix = computeSofaPathPrefix();
    return prefix;
}

std::map<std::string, std::string> Utils::readBasicIniFile(const std::string& path)
{
    std::map<std::string, std::string> map;
    std::ifstream iniFile(path.c_str());
    if (!iniFile.good())
    {
        Logger::getMainLogger().log(Logger::Error, "Error while trying to read file (" + path + ")", "Utils::readBasicIniFile()");
    }

    std::string line;
    while (std::getline(iniFile, line))
    {
        size_t equalPos = line.find_first_of('=');
        if (equalPos != std::string::npos)
        {
            const std::string key = line.substr(0, equalPos);
            const std::string value = line.substr(equalPos + 1, std::string::npos);
            map[key] = value;
        }
    }

    return map;
}


} // namespace helper
} // namespace sofa

