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
#include <sofa/helper/system/Utils.h>

#ifdef WIN32
# include <Windows.h>
# include <StrSafe.h>
#elif defined _XBOX
# include <xtl.h>
#elif defined __APPLE__
# include <mach-o/dyld.h>       // for _NSGetExecutablePath()
#else
# include <unistd.h>            // for readlink()
# include <errno.h>
# include <linux/limits.h>      // for PATH_MAX
#endif

#include <vector>
#include <iostream>

namespace sofa
{
namespace helper
{
namespace system
{


namespace Utils
{

#if defined WIN32 || defined _XBOX
std::wstring s2ws(const std::string& s)
{
    int len;
    int slength = (int)s.length() + 1;
    len = MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, 0, 0);
    wchar_t* buf = new wchar_t[len];
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), slength, buf, len);
    std::wstring r(buf);
    delete[] buf;
    return r;
}

std::string ws2s(const std::wstring& ws)
{
    // This is terrible, it will truncate wchar to char,
    // but it should work for ASCII characters.
    return std::string(ws.begin(), ws.end());
}

# ifdef WIN32
std::string GetLastError() {
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
    return ws2s(wsMessage);
}
# else  // XBOX
std::string GetLastError() {
    DWORD dwErrorCode = ::GetLastError();
    char buffer[32];
    sprintf_s(buffer, 32, "0x%08.8X", dwErrorCode);
    return buffer;
}
# endif
#endif

std::string getExecutablePath() {

#if defined(_XBOX) || defined(PS3)
    std::cerr << "Error: Utils::getExecutablePath() is not implemented." << std::endl;
    return "";

#elif defined(WIN32)
    std::vector<TCHAR> lpFilename(MAX_PATH);
    int ret = GetModuleFileName(NULL, /* NULL --> executable of the current process */
        &lpFilename[0],
        MAX_PATH);
    if (ret == 0 || ret == MAX_PATH) {
        std::cerr << "Utils::getExecutablePath(): " << GetLastError() << std::endl;
        return "";
    } else {
        return ws2s(std::wstring(&lpFilename[0]));
    }

#elif defined(__APPLE__)
    std::vector<char> path(PATH_MAX);
    std::vector<char> real_path(PATH_MAX);
    uint32_t size = path.size();
    if (_NSGetExecutablePath(&path[0], &size) != 0) {
        std::cerr << "Utils::getExecutablePath(): _NSGetExecutablePath() failed" << std::endl;
        return "";
    }
    if (realpath(&path[0], &real_path[0]) == 0) {
        std::cerr << "Utils::getExecutablePath(): realpath() failed" << std::endl;
        return "";
    }
    return std::string(&real_path[0]);

#else  // Linux
    std::vector<char> buffer(PATH_MAX);
    if (readlink("/proc/self/exe", &buffer[0], buffer.size()) == -1) {
        int error = errno;
        std::cerr << "Utils::getExecutablePath(): " << strerror(error) << std::endl;
        return "";
    } else {
        return std::string(&buffer[0]);
    }
#endif
}


} // namespace Utils


} // namespace system
} // namespace helper
} // namespace sofa

