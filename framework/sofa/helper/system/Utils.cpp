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
# include <windows.h>
# include <StrSafe.h>
#endif

namespace sofa
{
namespace helper
{
namespace system
{


namespace Utils
{

#ifdef WIN32
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
#endif

} // namespace Utils


} // namespace system
} // namespace helper
} // namespace sofa

