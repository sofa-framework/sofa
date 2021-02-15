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

#include <boost/filesystem.hpp>
#include <fstream>

using sofa::helper::Utils;

namespace sofa
{
namespace helper
{
namespace system
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
        if(loadPlugin(plugin))
        {
            Plugin* p = getPlugin(plugin);
            if(p) // should always be true as we are protected by if(loadPlugin(...))
            {
                p->initExternalModule();
            }
        }
    }
    instream.close();
}

void PluginManager::writeToIniFile(const std::string& path)
{
    std::ofstream outstream(path.c_str());
    PluginIterator iter;
    for( iter = m_pluginMap.begin(); iter!=m_pluginMap.end(); ++iter)
    {
        const std::string& pluginPath = (iter->first);
        outstream << pluginPath << " ";
        outstream << getPlugin(pluginPath)->getModuleVersion() << "\n";
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

bool PluginManager::loadPluginByPath(const std::string& pluginPath, std::ostream* errlog)
{
    if (pluginIsLoaded(pluginPath))
    {
        return true;
    }

    if (!FileSystem::exists(pluginPath))
    {
        const std::string msg = "File not found: " + pluginPath;
        msg_error("PluginManager") << msg;
        if (errlog) (*errlog) << msg << std::endl;
        return false;
    }

    DynamicLibrary::Handle d  = DynamicLibrary::load(pluginPath);
    Plugin p;
    if( ! d.isValid() )
    {
        const std::string msg = "Plugin loading failed (" + pluginPath + "): " + DynamicLibrary::getLastError();
        msg_error("PluginManager") << msg;
        if (errlog) (*errlog) << msg << std::endl;
        return false;
    }
    else
    {
        if(! getPluginEntry(p.initExternalModule,d))
        {
            const std::string msg = "Plugin loading failed (" + pluginPath + "): function initExternalModule() not found";
            msg_error("PluginManager") << msg;
            if (errlog) (*errlog) << msg << std::endl;
            return false;
        }
        getPluginEntry(p.getModuleName,d);

        if (checkDuplicatedPlugin(p, pluginPath))
        {
            return true;
        }

        getPluginEntry(p.getModuleDescription,d);
        getPluginEntry(p.getModuleLicense,d);
        getPluginEntry(p.getModuleComponentList,d);
        getPluginEntry(p.getModuleVersion,d);
    }

    p.dynamicLibrary = d;
    m_pluginMap[pluginPath] = p;
    p.initExternalModule();

    msg_info("PluginManager") << "Loaded plugin: " << pluginPath;

    for (const auto& [key, callback] : m_onPluginLoadedCallbacks)
    {
        if(callback)
        {
            callback(pluginPath, p);
        }
    }

    return true;
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

bool PluginManager::loadPluginByName(const std::string& pluginName, const std::string& suffix, bool ignoreCase, bool recursive, std::ostream* errlog)
{
    std::string pluginPath = findPlugin(pluginName, suffix, ignoreCase, recursive);

    if (!pluginPath.empty())
    {
        return loadPluginByPath(pluginPath, errlog);
    }
    else
    {
        const std::string msg = "Plugin not found: \"" + pluginName + suffix + "\"";
        if (errlog) (*errlog) << msg << std::endl;
        else msg_error("PluginManager") << msg;

        return false;
    }
}

bool PluginManager::loadPlugin(const std::string& plugin, const std::string& suffix, bool ignoreCase, bool recursive, std::ostream* errlog)
{
    if (FileSystem::isFile(plugin))
    {
        return loadPluginByPath(plugin,  errlog);
    }
    else
    {
        return loadPluginByName(plugin, suffix, ignoreCase, recursive, errlog);
    }
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
    std::string pluginPath = plugin;

    if (!FileSystem::isFile(plugin)) {
        return getPluginByName(plugin);
    }

    if (!pluginPath.empty() && m_pluginMap.find(pluginPath) != m_pluginMap.end())
    {
        return &m_pluginMap[pluginPath];
    }
    else
    {
        msg_info("PluginManager") << "Plugin not found in loaded plugins: " << plugin;
        return nullptr;
    }
}

Plugin* PluginManager::getPluginByName(const std::string& pluginName)
{
    for (PluginMap::iterator itP = m_pluginMap.begin(); itP != m_pluginMap.end(); ++itP)
    {
        std::string name(itP->second.getModuleName());
        if (name.compare(pluginName) == 0)
        {
            return &itP->second;
        }
    }

    msg_info("PluginManager") << "Plugin not found in loaded plugins: " << pluginName;
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
    PluginMap::iterator iter = m_pluginMap.find(pluginPath);
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
    for (std::vector<std::string>::iterator i = searchPaths.begin(); i!=searchPaths.end(); i++)
    {
        const std::string path = *i + "/" + libName;
        if (FileSystem::isFile(path))
            return path;
    }
    // Second try: case insensitive and recursive
    if (ignoreCase)
    {
        if(!recursive) maxRecursiveDepth = 0;
        const std::string downcaseLibName = Utils::downcaseString(libName);

        for (std::vector<std::string>::iterator i = searchPaths.begin(); i!=searchPaths.end(); i++)
        {
            const std::string& dir = *i;

            boost::filesystem::recursive_directory_iterator iter(dir);
            boost::filesystem::recursive_directory_iterator end;

            while (iter != end)
            {
                if ( iter.level() > maxRecursiveDepth )
                {
                    iter.no_push(); // skip
                }
                else if ( !boost::filesystem::is_directory(iter->path()) )
                {
                    const std::string path = iter->path().string();
                    const std::string filename = iter->path().filename().string();
                    const std::string downcaseFilename = Utils::downcaseString(filename);

                    if (downcaseFilename == downcaseLibName)
                    {
                        return FileSystem::cleanPath(path);
                    }
                }

                boost::system::error_code ec;
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
    std::string pluginPath = plugin;

    /// If we are not providing a filename then we have either to iterate in the plugin
    /// map to check no plugin has the same name or check in there is no accessible path
    /// in the plugin repository matching the pluginName
    if (!FileSystem::isFile(plugin))
    {
        /// Here is the iteration in the loaded plugin map
        for(auto k : m_pluginMap)
        {
            if(plugin == k.second.getModuleName())
                return true;
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

}

}
