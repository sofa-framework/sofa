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
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/Locale.h>
#include <sofa/helper/system/FileRepository.h>
#include <algorithm>

#ifdef WIN32
# include <Windows.h>
# include <StrSafe.h>
#elif defined __APPLE__
# include <mach-o/dyld.h>       // for _NSGetExecutablePath()
# include <errno.h>
#else
# include <string.h>            // for strerror()
# include <unistd.h>            // for readlink()
# include <errno.h>
# include <linux/limits.h>      // for PATH_MAX
#endif

#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

#include <sofa/helper/logging/Messaging.h>


using sofa::helper::system::FileSystem;

namespace sofa
{
namespace helper
{

std::wstring Utils::widenString(const std::string& s)
{
    // Set LC_CTYPE according to the environnement variable, for mbsrtowcs().
    system::TemporaryLocale locale(LC_CTYPE, "");

    const char * src = s.c_str();
    // Call mbsrtowcs() once to find out the length of the converted string.
    size_t length = mbsrtowcs(nullptr, &src, 0, nullptr);
    if (length == size_t(-1)) {
        const int error = errno;
        msg_warning("Utils::widenString()") << strerror(error);
        return L"";
    }

    // Call mbsrtowcs() again with a correctly sized buffer to actually do the conversion.
    wchar_t * buffer = new wchar_t[length + 1];
    length = mbsrtowcs(buffer, &src, length + 1, nullptr);
    if (length == size_t(-1)) {
        const int error = errno;
        msg_warning("Utils::widenString()") << strerror(error);
        delete[] buffer;
        return L"";
    }

    if (src != nullptr) {
        msg_warning("Utils::widenString()") << "Conversion failed (\"" << s << "\")";
        delete[] buffer;
        return L"";
    }

    std::wstring result(buffer);
    delete[] buffer;
    return result;
}


std::string Utils::narrowString(const std::wstring& ws)
{
    // Set LC_CTYPE according to the environnement variable, for wcstombs().
    system::TemporaryLocale locale(LC_CTYPE, "");

    const wchar_t * src = ws.c_str();
    // Call wcstombs() once to find out the length of the converted string.
    size_t length = wcstombs(nullptr, src, 0);
    if (length == size_t(-1)) {
        msg_warning("Utils::narrowString()") << "Conversion failed";
        return "";
    }

    // Call wcstombs() again with a correctly sized buffer to actually do the conversion.
    char * buffer = new char[length + 1];
    length = wcstombs(buffer, src, length + 1);
    if (length == size_t(-1)) {
        msg_warning("Utils::narrowString()") << "Conversion failed";
        delete[] buffer;
        return "";
    }

    std::string result(buffer);
    delete[] buffer;
    return result;
}


std::string Utils::downcaseString(const std::string& s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}


std::string Utils::upcaseString(const std::string& s)
{
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    return result;
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
    return narrowString(wsMessage);
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
        path = Utils::narrowString(std::wstring(&lpFilename[0]));
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


} // namespace helper
} // namespace sofa

