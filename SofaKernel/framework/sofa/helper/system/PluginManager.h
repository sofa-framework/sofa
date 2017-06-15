/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_HELPER_SYSTEM_PLUGINMANAGER_H
#define SOFA_HELPER_SYSTEM_PLUGINMANAGER_H

#include <sofa/helper/helper.h>
#include <sofa/helper/system/DynamicLibrary.h>
#include <vector>
#include <map>
#include <memory>

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
        const char* operator() () const
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
        const char* operator() () const
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
        const char* operator() () const
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
        const char* operator() () const
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
        const char* operator() () const
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
    DynamicLibrary::Handle dynamicLibrary;

};

class SOFA_HELPER_API PluginManager
{
public:
    typedef std::map<std::string, Plugin > PluginMap;
    typedef PluginMap::iterator PluginIterator;

    static PluginManager& getInstance();
    bool loadPlugin(const std::string& plugin, std::ostream* errlog=NULL);
    bool loadPluginByPath(const std::string& path, std::ostream* errlog=NULL);
    bool loadPluginByName(const std::string& pluginName, std::ostream* errlog=NULL);
    bool unloadPlugin(const std::string& path, std::ostream* errlog=NULL);

    void init();
	void init(const std::string& pluginPath);

    std::string findPlugin(const std::string& pluginName, bool ignoreCase = true);
    bool pluginIsLoaded(const std::string& pluginPath);

    inline friend std::ostream& operator<< ( std::ostream& os, const PluginManager& pluginManager )
    {
        return pluginManager.writeToStream( os );
    }
    inline friend std::istream& operator>> ( std::istream& in, PluginManager& pluginManager )
    {
        return pluginManager.readFromStream( in );
    }

    PluginMap& getPluginMap()  { return m_pluginMap; }
    std::vector<std::string>& getSearchPaths() { return m_searchPaths; }

    void readFromIniFile(const std::string& path);
    void writeToIniFile(const std::string& path);

    static std::string s_gui_postfix; ///< the postfix to gui plugin, default="gui" (e.g. myplugin_gui.so)

private:
    PluginManager();
    ~PluginManager();
    PluginManager(const PluginManager& );
    std::ostream& writeToStream( std::ostream& ) const;
    std::istream& readFromStream( std::istream& );
private:
    PluginMap m_pluginMap;
    std::vector<std::string> m_searchPaths;
};


}

}

}

#endif //SOFA_HELPER_SYSTEM_PLUGINMANAGER_H

