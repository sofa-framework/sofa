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
#ifndef SOFA_HELPER_SYSTEM_PLUGINMANAGER_H
#define SOFA_HELPER_SYSTEM_PLUGINMANAGER_H

#include <sofa/helper/config.h>
#include <sofa/helper/system/DynamicLibrary.h>
#include <sofa/helper/logging/Messaging.h>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <unordered_set>
#include <sofa/type/vector.h>

namespace sofa::helper::system
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
        InitExternalModule():func(nullptr) {}
    } InitExternalModule;

    typedef struct GetModuleName
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() () const
        {
            if (func) return func();
            else return nullptr;
        }
        GetModuleName():func(nullptr) {}
    } GetModuleName;

    typedef struct GetModuleDescription
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() () const
        {
            if (func) return func();
            else return nullptr;
        }
        GetModuleDescription():func(nullptr) {}
    } GetModuleDescription;

    typedef struct GetModuleLicense
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() () const
        {
            if (func) return func();
            else return nullptr;
        }

        GetModuleLicense():func(nullptr) {}
    } GetModuleLicense;

    typedef struct GetModuleComponentList
    {
        static  const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;

        SOFA_ATTRIBUTE_DEPRECATED__PLUGIN_GETCOMPONENTLIST()
        const char* operator() () const
        {
            if (func)
            {
                msg_warning("Plugin::GetModuleComponentList") << "This entrypoint is being deprecated, and should not be implemented anymore.";
                return func();
            }
            else return nullptr;
        }
        GetModuleComponentList():func(nullptr) {}
    } GetModuleComponentList;

    typedef struct GetModuleVersion
    {
        static const char* symbol;
        typedef const char* (*FuncPtr) ();
        FuncPtr func;
        const char* operator() () const
        {
            if (func) return func();
            else return nullptr;
        }
        GetModuleVersion():func(nullptr) {}
    } GetModuleVersion;

    struct ModuleIsInitialized
    {
        static const char* symbol;
        typedef bool (*FuncPtr) ();
        FuncPtr func;
        bool operator() () const
        {
            return (func) ? func() : false;
        }
        ModuleIsInitialized() :func(nullptr) {}
    };

    InitExternalModule     initExternalModule;
    GetModuleName          getModuleName;
    GetModuleDescription   getModuleDescription;
    GetModuleLicense       getModuleLicense;
    GetModuleComponentList getModuleComponentList;
    GetModuleVersion       getModuleVersion;
    ModuleIsInitialized    moduleIsInitialized;
private:
    DynamicLibrary::Handle dynamicLibrary;

};

namespace
{
    template <class LibraryEntry>
    [[nodiscard]] static bool getPluginEntry(LibraryEntry& entry, DynamicLibrary::Handle handle)
    {
        typedef typename LibraryEntry::FuncPtr FuncPtr;
        entry.func = (FuncPtr)DynamicLibrary::getSymbolAddress(handle, entry.symbol);
        return entry.func != 0;
    }
}

class SOFA_HELPER_API PluginManager
{
public:
    /// Map to store the list of plugin registered, key is the plugin path
    typedef std::map<std::string, Plugin > PluginMap;
    typedef PluginMap::iterator PluginIterator;

    static PluginManager& getInstance();
    /// Get the default suffix applied to plugin names to find the actual lib to load
    /// Returns "_d" in debug configuration and an empty string otherwise 
    static std::string getDefaultSuffix();

    enum class PluginLoadStatus : unsigned char
    {
        SUCCESS,
        ALREADY_LOADED,
        PLUGIN_FILE_NOT_FOUND,
        INVALID_LOADING,
        MISSING_SYMBOL,
        INIT_ERROR
    };

    
    /// Loads a plugin library in process memory and register into the map
    /// - if already registered into the map (and therefore loaded in memory), do nothing.
    /// - If not registered but loaded in memory, call entrypoints and register into the map
    /// - If not registered and not loaded in memory, it will load the plugin in memory, call entrypoints and register into the map
    /// @param plugin Can be just the filename of the library to load (without extension) or the full path
    /// @param suffix An optional suffix to apply to the filename. Defaults to "_d" with debug builds and is empty otherwise.
    /// @param ignoreCase Specify if the plugin search should be case-insensitive (activated by default).
    ///                   Not used if the plugin string passed as a parameter is a full path
    /// @param errlog An optional stream for error logging.
    PluginLoadStatus loadPlugin(const std::string& plugin, const std::string& suffix = getDefaultSuffix(), bool ignoreCase = true, bool recursive = true, std::ostream* errlog = nullptr);

    /// Loads a plugin library in process memory. 
    /// @param path The full path of the plugin to load
    /// @param errlog An optional stream for error logging.
    PluginLoadStatus loadPluginByPath(const std::string& path, std::ostream* errlog= nullptr);
    
    /// Loads a plugin library in process memory. 
    /// @param pluginName The filename without extension of the plugin to load
    /// @param suffix An optional suffix to apply to the filename. Defaults to "_d" with debug builds, empty otherwise.
    /// @param ignoreCase Specify if the plugin search should be case-insensitive (activated by default).
    ///                   Not used if the plugin string passed as a parameter is a full path
    /// @param errlog An optional stream for error logging.
    PluginLoadStatus loadPluginByName(const std::string& pluginName, const std::string& suffix = getDefaultSuffix(), bool ignoreCase = true, bool recursive = true, std::ostream* errlog= nullptr);
    
    /// Unloads a plugin from the map
    /// Warning: a previously loaded plugin will always be in process memory.
    bool unloadPlugin(const std::string& path, std::ostream* errlog= nullptr);

    /// Register a plugin. Merely an alias for loadPlugin()
    PluginLoadStatus registerPlugin(const std::string& plugin, const std::string& suffix = getDefaultSuffix(), bool ignoreCase = true, bool recursive = true, std::ostream* errlog = nullptr);

    [[nodiscard]] const std::unordered_set<std::string>& unloadedPlugins() const;

    [[nodiscard]] bool isPluginUnloaded(const std::string& pluginName) const;

    void init();
    void init(const std::string& pluginPath);
    void cleanup();

    std::string findPlugin(const std::string& pluginName, const std::string& suffix = getDefaultSuffix(), bool ignoreCase = true, bool recursive = true, int maxRecursiveDepth = 3);
    bool pluginIsLoaded(const std::string& plugin);

    /**
     * Determine if a plugin name or plugin path is known from the plugin
     * manager (i.e. has been loaded by the plugin manager) with the found path.
     * @param plugin A path to a plugin or a plugin name
     * @return A pair consisting of the found plugin path (or the plugin path
     * that was last tried) and a bool value set to true if the plugin has been
     * found in the plugin registration map
     */
    std::pair<std::string, bool> isPluginLoaded(const std::string& plugin);

    bool checkDuplicatedPlugin(const Plugin& plugin, const std::string& pluginPath);

    inline friend std::ostream& operator<< ( std::ostream& os, const PluginManager& pluginManager )
    {
        return pluginManager.writeToStream( os );
    }
    inline friend std::istream& operator>> ( std::istream& in, PluginManager& pluginManager )
    {
        return pluginManager.readFromStream( in );
    }

    PluginMap& getPluginMap()  { return m_pluginMap; }

    Plugin* getPlugin(const std::string& plugin, const std::string& = getDefaultSuffix(), bool = true);
    Plugin* getPluginByName(const std::string& pluginName);

    template <typename Entry>
    bool getEntryFromPlugin(const Plugin* plugin, Entry& entry)
    {
        return getPluginEntry(entry, plugin->dynamicLibrary);
    }

    void readFromIniFile(const std::string& path);
    void readFromIniFile(const std::string& path, type::vector<std::string>& listLoadedPlugins);
    void writeToIniFile(const std::string& path);

    static std::string s_gui_postfix; ///< the postfix to gui plugin, default="gui" (e.g. myplugin_gui.so)

    void addOnPluginLoadedCallback(const std::string& key, std::function<void(const std::string&, const Plugin&)> callback);
    void addOnPluginCleanupCallbacks(const std::string& key, std::function<void()> callback);
    void removeOnPluginLoadedCallback(const std::string& key);
    void removeOnPluginCleanupCallbacks(const std::string& key);

    static std::string GetPluginNameFromPath(const std::string& pluginPath);

private:
    PluginManager();
    ~PluginManager();
    PluginManager(const PluginManager& );
    std::ostream& writeToStream( std::ostream& ) const;
    std::istream& readFromStream( std::istream& );

    PluginMap m_pluginMap;
    std::map<std::string, std::function<void(const std::string&, const Plugin&)>> m_onPluginLoadedCallbacks;
    std::map<std::string, std::function<void()>> m_onPluginCleanupCallbacks;

    // contains the list of plugin names that were unloaded
    std::unordered_set<std::string> m_unloadedPlugins;
};


}

#endif //SOFA_HELPER_SYSTEM_PLUGINMANAGER_H

