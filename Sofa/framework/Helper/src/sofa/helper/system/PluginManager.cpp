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
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileSystem.h>
using sofa::helper::system::FileSystem;
#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>

#if __has_include(<filesystem>)
  #include <filesystem>
  namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
  #include <experimental/filesystem> 
  namespace fs = std::experimental::filesystem;
#else
  error "Missing the <filesystem> header."
#endif

#include <fstream>
#include <array>

using sofa::helper::Utils;

namespace sofa::helper::system
{

namespace
{

template <class LibraryEntry>
bool getPluginEntry(LibraryEntry& entry, DynamicLibrary::Handle handle)
{
    typedef typename LibraryEntry::FuncPtr FuncPtr;
    entry.func = (FuncPtr)DynamicLibrary::getSymbolAddress(handle, entry.symbol);
    if( entry.func == 0 )
    {
        return false;
    }
    else
    {
        return true;
    }
}

} // namespace

const char* Plugin::GetModuleComponentList::symbol    = "getModuleComponentList";
const char* Plugin::InitExternalModule::symbol        = "initExternalModule";
const char* Plugin::GetModuleDescription::symbol      = "getModuleDescription";
const char* Plugin::GetModuleLicense::symbol          = "getModuleLicense";
const char* Plugin::GetModuleName::symbol             = "getModuleName";
const char* Plugin::GetModuleVersion::symbol          = "getModuleVersion";
const char* Plugin::ModuleIsInitialized::symbol       = "moduleIsInitialized";

std::string PluginManager::s_gui_postfix = "gui";

PluginManager & PluginManager::getInstance()
{
    static PluginManager instance;
    return instance;
}

PluginManager::PluginManager()
{
}

PluginManager::~PluginManager()
{
    // BUGFIX: writeToIniFile should not be called here as it will erase the file in case it was not loaded
    // Instead we write the file each time a change have been made in the GUI and should be saved
    //writeToIniFile();
}

void PluginManager::readFromIniFile(const std::string& path)
{
    type::vector<std::string> loadedPlugins;
    readFromIniFile(path, loadedPlugins);
}

void PluginManager::readFromIniFile(const std::string& path, type::vector<std::string>& listLoadedPlugins)
{
    std::ifstream instream(path.c_str());
    std::string plugin, line, version;
    while(std::getline(instream, line))
    {
        if (line.empty()) continue;

        std::istringstream is(line);
        is >> plugin;
        if (is.eof())
            msg_deprecated("PluginManager") << path << " file is using a deprecated syntax (version information missing). Please update it in the near future.";
        else
            is >> version; // information not used for now
        if (loadPlugin(plugin) == PluginLoadStatus::SUCCESS)
        {
            Plugin* p = getPlugin(plugin);
            if(p) // should always be true as we are protected by if(loadPlugin(...))
            {
                p->initExternalModule();
                listLoadedPlugins.push_back(plugin);
            }
        }
    }
    instream.close();
    msg_info("PluginManager") << listLoadedPlugins.size() << " plugins have been loaded from " << path;
}

void PluginManager::writeToIniFile(const std::string& path)
{
    std::ofstream outstream(path.c_str());
    for( const auto& [pluginPath, _] : m_pluginMap)
    {
        if (const auto* plugin = getPlugin(pluginPath))
        {
            outstream << pluginPath;
            if (const char* moduleVersion = plugin->getModuleVersion())
            {
                outstream << " " << moduleVersion;
            }
            else
            {
                msg_error("PluginManager") << "The module '" << pluginPath << "' did not provide a module version";
                outstream << " <missingModuleVersion>";
            }
            outstream << "\n";
        }
        else
        {
            msg_error("PluginManager") << "Cannot find a valid plugin from path '" << pluginPath << "'";
        }
    }
    outstream.close();
}

/// Get the default suffix applied to plugin names to find the actual lib to load
/// (depends on platform, version, debug/release build)
std::string PluginManager::getDefaultSuffix()
{
#ifdef SOFA_LIBSUFFIX
    return sofa_tostring(SOFA_LIBSUFFIX);
#else
    return std::string();
#endif
}

PluginManager::PluginLoadStatus PluginManager::loadPluginByPath(const std::string& pluginPath, std::ostream* errlog)
{
    if (pluginIsLoaded(pluginPath))
    {
        const std::string msg = "Plugin '" + pluginPath + "' is already loaded";
        if (errlog) (*errlog) << msg << std::endl;
        return PluginLoadStatus::ALREADY_LOADED;
    }

    if (!FileSystem::exists(pluginPath))
    {
        const std::string msg = "File not found: " + pluginPath;
        msg_error("PluginManager") << msg;
        if (errlog) (*errlog) << msg << std::endl;
        return PluginLoadStatus::PLUGIN_FILE_NOT_FOUND;
    }

    const DynamicLibrary::Handle d  = DynamicLibrary::load(pluginPath);
    Plugin p;
    if( ! d.isValid() )
    {
        const std::string msg = "Plugin loading failed (" + pluginPath + "): " + DynamicLibrary::getLastError();
        msg_error("PluginManager") << msg;
        if (errlog) (*errlog) << msg << std::endl;
        return PluginLoadStatus::INVALID_LOADING;
    }
    else
    {
        if(! getPluginEntry(p.initExternalModule,d))
        {
            const std::string msg = "Plugin loading failed (" + pluginPath + "): function initExternalModule() not found";
            msg_error("PluginManager") << msg;
            if (errlog) (*errlog) << msg << std::endl;
            return PluginLoadStatus::MISSING_SYMBOL;
        }
        getPluginEntry(p.getModuleName,d);

        if (checkDuplicatedPlugin(p, pluginPath))
        {
            return PluginLoadStatus::ALREADY_LOADED;
        }

        getPluginEntry(p.getModuleDescription,d);
        getPluginEntry(p.getModuleLicense,d);
        getPluginEntry(p.getModuleComponentList,d);
        getPluginEntry(p.getModuleVersion,d);
    }

    p.dynamicLibrary = d;
    m_pluginMap[pluginPath] = p;
    p.initExternalModule();

    // check if the plugin is initialized (if it can report this information)
    if (getPluginEntry(p.moduleIsInitialized, d))
    {
        if (!p.moduleIsInitialized())
        {
            const std::string msg = pluginPath + " reported an error while trying to initialize. This plugin will not be loaded.";
            msg_error("PluginManager") << msg;
            if (errlog) (*errlog) << msg << std::endl;

            unloadPlugin(pluginPath);
            return PluginLoadStatus::INIT_ERROR;
        }
    }

    msg_info("PluginManager") << "Loaded plugin: " << pluginPath;

    for (const auto& [key, callback] : m_onPluginLoadedCallbacks)
    {
        if(callback)
        {
            callback(pluginPath, p);
        }
    }

    return PluginLoadStatus::SUCCESS;
}

void PluginManager::addOnPluginLoadedCallback(const std::string& key, std::function<void(const std::string&, const Plugin&)> callback)
{
    if(m_onPluginLoadedCallbacks.find(key) == m_onPluginLoadedCallbacks.end())
    {
        m_onPluginLoadedCallbacks[key] = callback;
    }
}

void PluginManager::removeOnPluginLoadedCallback(const std::string& key)
{
    m_onPluginLoadedCallbacks.erase(key);
}

std::string PluginManager::GetPluginNameFromPath(const std::string& pluginPath)
{
    const auto filename = sofa::helper::system::SetDirectory::GetFileName(pluginPath.c_str());
    const std::string::size_type pos = filename.find_last_of("." + DynamicLibrary::extension);
    if (pos != std::string::npos)
    {
        return filename.substr(0,pos);
    }

    return sofa::helper::system::SetDirectory::GetFileNameWithoutExtension(pluginPath.c_str());;
}

auto PluginManager::loadPluginByName(const std::string& pluginName, const std::string& suffix, bool ignoreCase,
                                     bool recursive, std::ostream* errlog) -> PluginLoadStatus
{
    const std::string pluginPath = findPlugin(pluginName, suffix, ignoreCase, recursive);

    if (!pluginPath.empty())
    {
        return loadPluginByPath(pluginPath, errlog);
    }

    const std::string msg = "Plugin not found: \"" + pluginName + suffix + "\"";
    if (errlog)
    {
        (*errlog) << msg << std::endl;
    }
    else
    {
        msg_error("PluginManager") << msg;
    }

    return PluginLoadStatus::PLUGIN_FILE_NOT_FOUND;
}

auto PluginManager::loadPlugin(const std::string& plugin, const std::string& suffix, bool ignoreCase, bool recursive,
                               std::ostream* errlog) -> PluginLoadStatus
{
    if (FileSystem::isFile(plugin))
    {
        return loadPluginByPath(plugin,  errlog);
    }

    return loadPluginByName(plugin, suffix, ignoreCase, recursive, errlog);
}

bool PluginManager::unloadPlugin(const std::string &pluginPath, std::ostream* errlog)
{
    if(!pluginIsLoaded(pluginPath))
    {
        const std::string msg = "Plugin not loaded: " + pluginPath;
        msg_error("PluginManager::unloadPlugin()") << msg;
        if (errlog) (*errlog) << msg << std::endl;
        return false;
    }
    else
    {
        m_pluginMap.erase(m_pluginMap.find(pluginPath));
        removeOnPluginLoadedCallback(pluginPath);
        return true;
    }
}

Plugin* PluginManager::getPlugin(const std::string& plugin, const std::string& /*suffix*/, bool /*ignoreCase*/)
{
    const std::string pluginPath = plugin;

    if (!FileSystem::isFile(plugin)) {
        return getPluginByName(plugin);
    }

    if (!pluginPath.empty() && m_pluginMap.find(pluginPath) != m_pluginMap.end())
    {
        return &m_pluginMap[pluginPath];
    }
    else
    {
        // check if a plugin with a same name but a different path is loaded
        // problematic case per se but at least we can warn the user
        const auto& pluginName = GetPluginNameFromPath(pluginPath);
        for (auto& k : m_pluginMap)
        {
            if (pluginName == k.second.getModuleName())
            {
                msg_warning("PluginManager") << "Plugin " << pluginName << " is already loaded from a different path, check you configuration.";
                return &k.second;
            }
        }

        msg_warning("PluginManager") << "Plugin not found in loaded plugins: " << plugin;
        return nullptr;
    }
}

Plugin* PluginManager::getPluginByName(const std::string& pluginName)
{
    for (PluginMap::iterator itP = m_pluginMap.begin(); itP != m_pluginMap.end(); ++itP)
    {
        std::string name(itP->second.getModuleName());
        if (name == pluginName)
        {
            return &itP->second;
        }
    }

    msg_warning("PluginManager") << "Plugin not found in loaded plugins: " << pluginName;
    return nullptr;
}

std::istream& PluginManager::readFromStream(std::istream & in)
{
    while(!in.eof())
    {
        std::string pluginPath;
        in >> pluginPath;
        loadPlugin(pluginPath);
    }
    return in;
}

std::ostream& PluginManager::writeToStream(std::ostream & os) const
{
    PluginMap::const_iterator iter;
    for(iter= m_pluginMap.begin(); iter!=m_pluginMap.end(); ++iter)
    {
        os << iter->first;
    }
    return os;
}

void PluginManager::init()
{
    PluginMap::iterator iter;
    for( iter = m_pluginMap.begin(); iter!= m_pluginMap.end(); ++iter)
    {
        Plugin& plugin = iter->second;
        plugin.initExternalModule();
    }
}

void PluginManager::init(const std::string& pluginPath)
{
    const PluginMap::iterator iter = m_pluginMap.find(pluginPath);
    if(m_pluginMap.end() != iter)
    {
        Plugin& plugin = iter->second;
        plugin.initExternalModule();
    }
}



std::string PluginManager::findPlugin(const std::string& pluginName, const std::string& suffix, bool ignoreCase, bool recursive, int maxRecursiveDepth)
{
    std::vector<std::string> searchPaths = PluginRepository.getPaths();

    std::string name(pluginName);
    name  += suffix;
    const std::string libName = DynamicLibrary::prefix + name + "." + DynamicLibrary::extension;

    // First try: case sensitive
    for (const auto & prefix : searchPaths)
    {
        const std::array<std::string, 4> paths = {
            FileSystem::append(prefix, libName),
            FileSystem::append(prefix, pluginName, libName),
            FileSystem::append(prefix, pluginName, "bin", libName),
            FileSystem::append(prefix, pluginName, "lib", libName)
        };
        for (const auto & path : paths) {
            if (FileSystem::isFile(path)) {
                return path;
            }
        }
    }

    // Second try: case insensitive and recursive
    if (ignoreCase)
    {
        if(!recursive) maxRecursiveDepth = 0;
        const std::string downcaseLibName = Utils::downcaseString(libName);

        for (std::vector<std::string>::iterator i = searchPaths.begin(); i!=searchPaths.end(); i++)
        {
            const std::string& dir = *i;

            fs::recursive_directory_iterator iter(dir);
            fs::recursive_directory_iterator end;

            while (iter != end)
            {
                if ( iter.depth() > maxRecursiveDepth )
                {
                    iter.disable_recursion_pending(); // skip
                }
                else if ( !fs::is_directory(iter->path()) )
                {
                    const std::string path = iter->path().string();
                    const std::string filename = iter->path().filename().string();
                    const std::string downcaseFilename = Utils::downcaseString(filename);

                    if (downcaseFilename == downcaseLibName)
                    {
                        return FileSystem::cleanPath(path);
                    }
                }

                std::error_code ec;
                iter.increment(ec);
                if (ec)
                {
                    msg_error("PluginManager") << "Error while accessing " << iter->path().string() << ": " << ec.message();
                }
            }
        }
    }
    return std::string();
}

bool PluginManager::pluginIsLoaded(const std::string& plugin)
{
    if (plugin.empty()) return false;

    std::string pluginPath;

    /// If we are not providing a filename then we have either to iterate in the plugin
    /// map to check no plugin has the same name or check in there is no accessible path
    /// in the plugin repository matching the pluginName
    if (FileSystem::cleanPath(plugin, FileSystem::SLASH).find('/') != std::string::npos)
    {
        // plugin argument is a path
        if (!FileSystem::isFile(plugin))
        {
            // path is invalid
            msg_error("PluginManager") << "Could not check if the plugin is loaded as the path is invalid: " << plugin;
            return false;
        }

        pluginPath = plugin;

        // argument is a path but we need to check if it was not already loaded with a different path
        const auto& pluginName = GetPluginNameFromPath(pluginPath);
        for (const auto& [loadedPath, loadedPlugin] : m_pluginMap)
        {
            if (pluginName == loadedPlugin.getModuleName() && pluginPath != loadedPath)
            {
                // we did find a plugin with the same, but it does not have the same path...
                msg_warning("PluginManager") << "This plugin " << pluginName << " has been loaded from a different path, it will certainly lead to bugs or crashes... " << msgendl
                                             << "You tried to load: " << pluginPath << msgendl
                                             << "Already loaded: " << loadedPath;
                return true;
            }
        }

    }
    else
    {
        // plugin argument is a name
        /// Here is the iteration in the loaded plugin map
        for(auto k : m_pluginMap)
        {
            if(plugin == k.second.getModuleName())
            {
                return true;
            }
        }

        /// At this point we have not found a loaded plugin, we try to
        /// explore if the filesystem can help.
        pluginPath = findPlugin(plugin);
    }

    /// Check that the path (either provided by user or through the call to findPlugin()
    /// leads to a loaded plugin.
    return m_pluginMap.find(pluginPath) != m_pluginMap.end();
}

bool PluginManager::checkDuplicatedPlugin(const Plugin& plugin, const std::string& pluginPath)
{
    for (auto itP : m_pluginMap)
    {
        std::string name(itP.second.getModuleName());
        std::string plugName(plugin.getModuleName());
        if (name.compare(plugName) == 0 && pluginPath.compare(itP.first) != 0)
        {
            msg_warning("PluginManager") << "Trying to load plugin (" + name + ", from path: " + pluginPath + ") already registered from path: " + itP.first;
            return true;
        }
    }

    return false;
}

}
