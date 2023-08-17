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
#include <sofa/helper/system/DynamicLibrary.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;
#ifdef WIN32
# include <Windows.h>
#else
# include <dlfcn.h>
#endif
#include <string>

namespace sofa::helper::system
{


DynamicLibrary::Handle::Handle(const std::string& filename, void *handle)
    : m_realHandle(handle), m_filename(new std::string(filename))
{
}

DynamicLibrary::Handle::Handle(const Handle& that)
    : m_realHandle(that.m_realHandle), m_filename(that.m_filename)
{
}

DynamicLibrary::Handle::Handle(): m_realHandle(nullptr)
{
}

DynamicLibrary::Handle& DynamicLibrary::Handle::operator= (const Handle& other) {
    m_realHandle = other.m_realHandle;
    m_filename = other.m_filename;
    return *this;
}

bool DynamicLibrary::Handle::isValid() const
{
    return m_realHandle != nullptr;
}

const std::string& DynamicLibrary::Handle::filename() const
{
    return *m_filename;
}


DynamicLibrary::Handle DynamicLibrary::load(const std::string& filename)
{
# if defined(WIN32)
    const std::string p = FileSystem::cleanPath(filename, FileSystem::BACKSLASH);
    void *handle = ::LoadLibraryA(filename.c_str());
    if (handle == nullptr)
    {
        // Retry with altered search path
        handle = ::LoadLibraryExA(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
# else
    void *handle = ::dlopen(filename.c_str(), RTLD_NOW | RTLD_GLOBAL);
# endif
    if (handle == nullptr)
        fetchLastError();
    return handle ? Handle(filename, handle) : Handle();
}

int DynamicLibrary::unload(Handle handle)
{
# if defined(WIN32)
    int error = (::FreeLibrary((HMODULE)(handle.m_realHandle)) == 0);
# else
    int error = ::dlclose(handle.m_realHandle);
# endif
    if (error)
        fetchLastError();
    return error;
}

void * DynamicLibrary::getSymbolAddress(Handle handle,
                                        const std::string& symbol)
{
    if (!handle.isValid()) {
        m_lastError = "DynamicLibrary::getSymbolAddress(): invalid handle";
        return nullptr;
    }
# if defined(WIN32)
    void *symbolAddress = ::GetProcAddress((HMODULE)handle.m_realHandle,
                                     symbol.c_str());
# else
    void *symbolAddress = ::dlsym(handle.m_realHandle, symbol.c_str());
# endif
    if(symbolAddress == nullptr)
        fetchLastError();
    return symbolAddress;
}

std::string DynamicLibrary::getLastError()
{
    std::string msg = m_lastError;
    m_lastError = "";
    return msg;
}

void DynamicLibrary::fetchLastError()
{
#if defined(WIN32)
    LPTSTR pMsgBuf;
    FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER
                  | FORMAT_MESSAGE_FROM_SYSTEM
                  | FORMAT_MESSAGE_IGNORE_INSERTS,
                  nullptr, ::GetLastError(),
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (LPTSTR)&pMsgBuf, 0, nullptr);
# ifndef UNICODE
    m_lastError = std::string(pMsgBuf);
# else
    std::wstring s(pMsgBuf);
    // This is terrible, it will truncate wchar_t to char_t,
    // but it should work for characters 0 to 127.
    m_lastError = std::string(s.begin(), s.end());
# endif
    LocalFree(pMsgBuf);
#else
    const char *dlopenError = ::dlerror();
    m_lastError = dlopenError ? dlopenError : "";
#endif
}


#if defined(WIN32)
const std::string DynamicLibrary::extension = "dll";
#elif defined(__APPLE__)
const std::string DynamicLibrary::extension = "dylib";
#else
const std::string DynamicLibrary::extension = "so";
#endif

#if defined(WIN32)
const std::string DynamicLibrary::prefix = "";
#else
const std::string DynamicLibrary::prefix = "lib";
#endif


std::string DynamicLibrary::m_lastError = std::string("");


}
