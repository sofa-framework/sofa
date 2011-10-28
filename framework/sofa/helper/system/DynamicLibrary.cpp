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
