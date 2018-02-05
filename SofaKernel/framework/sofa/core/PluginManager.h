/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_CORE_PLUGINMANAGER_H
#define SOFA_CORE_PLUGINMANAGER_H

#include <sofa/core/core.h>
#include <sofa/helper/system/DynamicLibrary.h>
#include <sofa/helper/system/FileSystem.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace sofa
{
namespace core
{

class ObjectFactory;
class Plugin;

/**
 * This class manages a collection of plugins: loading, unloading,
 * initialization, etc
 */
class SOFA_CORE_API PluginManager
{
public:

    /// An entry corresponding to a plugin that in loaded in the PluginManager.
    struct SOFA_CORE_API LoadedPlugin {
        /// A pointer to the Plugin object.
        sofa::core::Plugin *plugin;
        /// The handle to the plugin's library
        sofa::helper::system::DynamicLibrary::Handle handle;
        LoadedPlugin();
        LoadedPlugin(Plugin *plugin, sofa::helper::system::DynamicLibrary::Handle handle);
    };

    PluginManager();
    PluginManager(const std::string& pluginDirectory);
    ~PluginManager();

    /// Add a directory where to look for plugins.
    void addPluginDirectory(const std::string& path);

    /// Try to load the given plugin; throw a runtime_error on error.
    ///
    /// @param plugin either the name of the plugin, or the path of the library
    /// @return A reference to the Plugin
    sofa::core::Plugin& loadPlugin(const std::string& plugin);

    /// Try to unload a plugin
    ///
    /// @return true if the plugin was unloaded. (A plugin can prevent unloading by overloading Plugin::canBeUnloaded())
    bool unloadPlugin(const std::string& pluginName);

    /// Unload all the loaded plugins
    void unloadAllPlugins();

    void addComponentsToFactory(sofa::core::ObjectFactory& factory,
                                const sofa::core::Plugin& plugin);

    void removeComponentsFromFactory(sofa::core::ObjectFactory& factory,
                                     const sofa::core::Plugin& plugin);

    void refreshPluginInfo();

    bool canFindComponent(std::string componentName, std::string templateParameters="");

    sofa::core::Plugin& loadPluginContaining(std::string componentName,
                                             std::string templateParameters="");

    /// A pair (template name, template parameters) or (class name, "") that
    /// identifies an instantiable component.
    ///
    /// In the case of a class template with empty template parameters, it
    /// refers to the default instanciation of this template.
    typedef std::pair<std::string, std::string> ComponentID;
    /// A map meant to index the names of the plugins by the ComponentID of each
    /// of their component.
    typedef std::map<ComponentID, std::string> ComponentDatabase;
    /// A map meant to store the currently loaded plugins, indexed by their name.
    typedef std::map<std::string, LoadedPlugin> LoadedPluginMap;

    /// Get the instance of a loaded Plugin from its name.
    const sofa::core::Plugin& getLoadedPlugin(const std::string& pluginName) const;
    const LoadedPluginMap& getLoadedPlugins() const;
    const LoadedPluginMap& getLoadedLegacyPlugins() const;
    const std::vector<std::string>& getPluginDirectories() const;
    const ComponentDatabase& getComponentDatabase() const;

protected:
    LoadedPlugin openPlugin(const std::string& filename);
    void closePlugin(LoadedPlugin entry);
    void processDirectory(std::string directory);

    /// The plugins currently loaded by this PluginManager, indexed by their
    /// name.
    std::map<std::string, LoadedPlugin> m_loadedPlugins;

    std::map<std::string, LoadedPlugin> m_loadedLegacyPlugins;

    /// The paths of the plugin libraries, indexed by the name of the plugin.
    std::map<std::string, std::string> m_pluginFiles;

    /// Description of the content of the plugins.
    ///
    /// It is a map of the plugin names, indexed by the ComponentID of the
    /// components they contain.
    ComponentDatabase m_componentDatabase;

    /// The paths to the plugin directories.
    std::vector<std::string> m_pluginDirectories;
};


}

}



#endif
