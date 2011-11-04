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
#include <sofa/helper/system/DynamicLibrary.h>
#ifdef WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#include <string>

namespace sofa
{
namespace helper
{
namespace system
{


DynamicLibrary::DynamicLibrary(const std::string& name, void * handle) : m_handle(handle)
    ,m_name(name)
{
}

DynamicLibrary::~DynamicLibrary()
{
    /* if (m_handle)
     {
     #ifndef WIN32
       ::dlclose(m_handle);
     #else
       ::FreeLibrary((HMODULE)m_handle);
     #endif
     } */
}

DynamicLibrary * DynamicLibrary::load(const std::string & name,
        std::ostream* errlog)
{
    if (name.empty())
    {
        (*errlog) <<  "Empty path." << std::endl;
        return NULL;
    }

    void * handle = NULL;

#ifdef WIN32
    handle = ::LoadLibraryA(name.c_str());
    if (handle == NULL)
    {
        DWORD errorCode = ::GetLastError();
        (*errlog) << "LoadLibrary("<<name<<") Failed. errorCode: "<<errorCode;
        (*errlog) << std::endl;
    }
#else
    handle = ::dlopen(name.c_str(), RTLD_NOW);
    if (!handle)
    {
        std::string dlErrorString;
        const char *zErrorString = ::dlerror();
        if (zErrorString)
            dlErrorString = zErrorString;
        (*errlog) <<  "Failed to load \"" + name + '"';
        if(dlErrorString.size())
            (*errlog) << ": " + dlErrorString;
        (*errlog) << std::endl;
        return NULL;
    }

#endif
    return new DynamicLibrary(name,handle);
}

void * DynamicLibrary::getSymbol(const std::string & symbol, std::ostream* errlog)
{
    if (!m_handle)
        return NULL;
    void* symbolAddress = NULL;

#ifdef WIN32
    symbolAddress =  ::GetProcAddress((HMODULE)m_handle, symbol.c_str());
#else
    symbolAddress =  ::dlsym(m_handle, symbol.c_str());
#endif
    if( symbolAddress == NULL )
    {
        (*errlog) << m_name << " symbol: "<< symbol <<" not found" << std::endl;
    }
    return symbolAddress;
}

const char* DynamicLibrary::getExtension()
{
#if defined(__APPLE__)
    return "dylib";
#elif defined(WIN32)
    return "dll";
#else
    return "so";
#endif

}

const char* DynamicLibrary::getSuffix()
{
#ifdef NDEBUG
    return "";
#else
    return "d";
#endif
}

}

}

}
