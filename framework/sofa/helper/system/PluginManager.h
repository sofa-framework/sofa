#ifndef SOFA_HELPER_SYSTEM_PLUGINMANAGER_H
#define SOFA_HELPER_SYSTEM_PLUGINMANAGER_H

#include <sofa/helper/system/DynamicLibrary.h>
#include <boost/shared_ptr.hpp>
#include <map>
#include <sofa/helper/helper.h>

namespace sofa
{
namespace helper
{
namespace system
{
class PluginManager;

class SOFA_HELPER_API Plugin
{
    friend class PluginManager;
public:
    typedef struct InitExternalModule
    {
        static const char* symbol;
        typedef void (*FuncPtr) ();
        FuncPtr func;
        void operator() ()
        {
            if (func) return func();
        }
        InitExternalModule():func(0) {}
    } InitExternalModule;

    typedef struct GetModuleName
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() ()
        {
            if (func) return func();
            else return NULL;
        }
        GetModuleName():func(0) {}
    } GetModuleName;

    typedef struct GetModuleDescription
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() ()
        {
            if (func) return func();
            else return NULL;
        }
        GetModuleDescription():func(0) {}
    } GetModuleDescription;

    typedef struct GetModuleLicense
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() ()
        {
            if (func) return func();
            else return NULL;
        }

        GetModuleLicense():func(0) {}
    } GetModuleLicense;

    typedef struct GetModuleComponentList
    {
        static  const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() ()
        {
            if (func) return func();
            else return NULL;
        }
        GetModuleComponentList():func(0) {}
    } GetModuleComponentList;

    typedef struct GetModuleVersion
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() ()
        {
            if (func) return func();
            else return NULL;
        }
        GetModuleVersion():func(0) {}
    } GetModuleVersion;

    InitExternalModule     initExternalModule;
    GetModuleName          getModuleName;
    GetModuleDescription   getModuleDescription;
    GetModuleLicense       getModuleLicense;
    GetModuleComponentList getModuleComponentList;
    GetModuleVersion       getModuleVersion;
private:
    boost::shared_ptr<DynamicLibrary> dynamicLibrary;

};

class SOFA_HELPER_API PluginManager
{
public:
    typedef std::map<const std::string, Plugin > PluginMap;
    typedef PluginMap::iterator PluginIterator;

    static PluginManager& getInstance();
    bool loadPlugin(const std::string& path, std::ostream* errlog=&std::cerr);
    bool unloadPlugin(const std::string& path, std::ostream* errlog=&std::cerr);

    void initRecentlyOpened();
    void init();

    inline friend std::ostream& operator<< ( std::ostream& os, const PluginManager& pluginManager )
    {
        return pluginManager.writeToStream( os );
    }
    inline friend std::istream& operator>> ( std::istream& in, PluginManager& pluginManager )
    {
        return pluginManager.readFromStream( in );
    }
    PluginMap& getPluginMap()  { return m_pluginMap; }



private:
    PluginManager() {}
    ~PluginManager();
    PluginManager(const PluginManager& );
    DynamicLibrary* loadLibrary(const std::string& path,  std::ostream* errlog=&std::cerr);
    std::ostream& writeToStream( std::ostream& ) const;
    std::istream& readFromStream( std::istream& );
    void readFromIniFile();
    void writeToIniFile();
private:
    PluginMap m_pluginMap;
};


}

}

}

#endif //SOFA_HELPER_SYSTEM_PLUGINMANAGER_H

