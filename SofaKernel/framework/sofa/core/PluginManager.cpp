/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <sofa/core/PluginManager.h>

#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/system/Utils.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/Plugin.h>

#include <exception>


using namespace sofa::helper::system;


namespace sofa
{
namespace core
{

PluginManager::PluginManager()
{
}

PluginManager::PluginManager(const std::string& pluginDirectory)
{
    addPluginDirectory(pluginDirectory);
#ifdef WIN32
	SetDllDirectory(Utils::s2ws(pluginDirectory).c_str());
#endif
}

PluginManager::LoadedPlugin::LoadedPlugin()
    : plugin(NULL)
{
}

PluginManager::LoadedPlugin::LoadedPlugin(Plugin *plugin_,
                                        DynamicLibrary::Handle handle_)
    : plugin(plugin_), handle(handle_)
{
}


PluginManager::~PluginManager()
{
    // unloadAllPlugins();
}

void PluginManager::unloadAllPlugins()
{
    size_t pluginsThatCantBeUnloaded = 0;
    while (m_loadedPlugins.size() != pluginsThatCantBeUnloaded) {
        const std::string pluginName(m_loadedPlugins.begin()->first);
        if (!unloadPlugin(pluginName))
            pluginsThatCantBeUnloaded++;
    }
}

void PluginManager::addPluginDirectory(const std::string& path)
{
    if (!FileSystem::exists(path))
        std::cerr << "PluginManager::addPluginDirectory(): error, path does not exist: "
                  << path << std::endl;
    else if (!FileSystem::isDirectory(path))
        std::cerr << "PluginManager::addPluginDirectory(): error, path is not a directory: "
                  << path << std::endl;
    else
        m_pluginDirectories.push_back(path);
}

PluginManager::LoadedPlugin PluginManager::openPlugin(const std::string& filename)
{
    // Open the dynamic library
    DynamicLibrary::Handle handle = DynamicLibrary::load(filename);
    if (!handle.isValid())
        throw std::runtime_error(
            "PluginManager: could not load dynamic library '" + filename + "':\n"
            + DynamicLibrary::getLastError());

    Plugin *plugin = NULL;
    // Retrieve the entry point (the 'get_plugin' function)
    void * get_plugin = DynamicLibrary::getSymbolAddress(handle, "get_plugin");
    if (get_plugin != NULL) {
        // Call the 'get_plugin' function to create an instance of Plugin
        plugin = (Plugin*) ((void*(*)(void))get_plugin)();
    } else {
        void * initExternalModule = DynamicLibrary::getSymbolAddress(handle, "initExternalModule");
        void * getModuleName = DynamicLibrary::getSymbolAddress(handle, "getModuleName");
        void * getModuleDescription = DynamicLibrary::getSymbolAddress(handle, "getModuleDescription");
        void * getModuleLicense = DynamicLibrary::getSymbolAddress(handle, "getModuleLicense");
        void * getModuleVersion = DynamicLibrary::getSymbolAddress(handle, "getModuleVersion");
        if (initExternalModule && getModuleName && getModuleDescription
            && getModuleLicense && getModuleVersion) {
            ((void(*)(void))initExternalModule)();
            plugin = new Plugin(((char*(*)(void))getModuleName)(),
                                ((char*(*)(void))getModuleDescription)(),
                                ((char*(*)(void))getModuleVersion)(),
                                ((char*(*)(void))getModuleLicense)(),
                                "", // authors
                                true);
        } else {
            DynamicLibrary::unload(handle);
            throw std::runtime_error("PluginManager: could find plugin entry point in '"
                                     + filename + "'\n" + DynamicLibrary::getLastError());

        }
    }
    return LoadedPlugin(plugin, handle);
}


void PluginManager::closePlugin(LoadedPlugin pluginEntry)
{
    // Close the library
    DynamicLibrary::unload(pluginEntry.handle);
}

Plugin& PluginManager::loadPlugin(const std::string& plugin)
{
    // If 'plugin' is a known plugin name, get its path, otherwise it
    // is considered to be a path
    const std::string pluginPath(m_pluginFiles.find(plugin) != m_pluginFiles.end() ?
                                 m_pluginFiles[plugin] :
                                 plugin);

    LoadedPlugin entry = openPlugin(pluginPath);
    const std::string pluginName(entry.plugin->getName());
    if (m_loadedPlugins.find(pluginName) != m_loadedPlugins.end()) {
        // If the plugin is already loaded, error
        closePlugin(entry);
        throw std::runtime_error("PluginManager: plugin already loaded: " + pluginName);
    } else {
        // Otherwise add the plugin to the collection
        if (entry.plugin->isLegacy()) {
            m_loadedLegacyPlugins[pluginName] = entry;
            std::cout << "PluginManager: loaded legacy plugin '" << pluginName << "'" << std::endl;
            return *m_loadedLegacyPlugins[pluginName].plugin;
        } else {
            m_loadedPlugins[pluginName] = entry;
            entry.plugin->init();
            std::cout << "PluginManager: loaded plugin '" << pluginName << "'" << std::endl;
            return *m_loadedPlugins[pluginName].plugin;
        }
    }

}

bool PluginManager::unloadPlugin(const std::string& pluginName)
{
    // If plugin not found, error
    if (m_loadedPlugins.find(pluginName) == m_loadedPlugins.end()) {
        throw std::runtime_error("plugin not found");
    }

    LoadedPlugin& entry = m_loadedPlugins[pluginName];
    if (entry.plugin->canBeUnloaded()) {
        if (!entry.plugin->exit())
            std::cout << "PluginManager: plugin '" << pluginName << "': error on exit()" << std::endl;
        closePlugin(entry);
        m_loadedPlugins.erase(pluginName);
        std::cout << "PluginManager: unloaded plugin '" << pluginName << "'" << std::endl;
        return true;
    } else {
        std::cout << "PluginManager: not unloading plugin '" << pluginName << "'" << std::endl;
        return false;
    }
}

void PluginManager::addComponentsToFactory(ObjectFactory& factory,
                                           const Plugin& plugin)
{
    // For each component
    for (std::map<std::string, Plugin::ComponentEntry>::const_iterator j =
             plugin.getComponentEntries().begin();
         j != plugin.getComponentEntries().end();
         j++) {
        const Plugin::ComponentEntry& component = j->second;

        if (!factory.hasEntry(component.name))
            factory.addEntry(component.name, component.description,
                             plugin.getAuthors(), plugin.getLicense());

        for (std::set<std::string>::const_iterator k = component.aliases.begin();
             k != component.aliases.end();
             k++) {
            factory.addAlias(*k, component.name);
        }

        for (ObjectFactory::CreatorMap::const_iterator k = component.creators.begin();
             k != component.creators.end();
             k++) {
            bool isDefault = component.defaultTemplateParameters == k->first;
            factory.addCreator(component.name, k->first, k->second, isDefault);
            // std::cout << "Added Creator for " << component.name << "<" << k->first << ">" << std::endl;
        }
    }
}

void PluginManager::removeComponentsFromFactory(ObjectFactory& factory,
                                                const Plugin& plugin)
{
    // For each component
    for (Plugin::ComponentEntryMap::const_iterator j = plugin.getComponentEntries().begin();
         j != plugin.getComponentEntries().end();
         j++) {
        const Plugin::ComponentEntry& component = j->second;

        // Remove each creator
        for (std::map<std::string, ObjectFactory::Creator::SPtr >::const_iterator k
                 = component.creators.begin();
             k != component.creators.end();
             k++) {
            factory.removeCreator(component.name, k->first);
            // std::cout << "Removed Creator for " << component.name << ", " <<k->first << std::endl;
        }
    }
}

void PluginManager::processDirectory(std::string directory)
{
    std::vector<std::string> files;
    FileSystem::listDirectory(directory, files, DynamicLibrary::extension);
    // For each alleged plugin
    for (std::size_t k = 0 ; k != files.size() ; k++) {
        const std::string filepath(directory + files[k]);
        try {
            // Try to open the library as a plugin
            LoadedPlugin pluginEntry = openPlugin(filepath);

            // For each component,
            const Plugin::ComponentEntryMap& componentMap
                = pluginEntry.plugin->getComponentEntries();
            for (Plugin::ComponentEntryMap::const_iterator i = componentMap.begin();
                 i != componentMap.end();
                 i++) {
                const Plugin::ComponentEntry& componentEntry = i->second;
                // for each creator,
                for (ObjectFactory::CreatorMap::const_iterator j = componentEntry.creators.begin();
                     j != componentEntry.creators.end();
                     j++) {
                    // remember the plugin's file
                    const ComponentID key(componentEntry.name, j->first);
                    m_componentDatabase[key] = pluginEntry.handle.filename();
                    if (j->first == componentEntry.defaultTemplateParameters) {
                        const ComponentID key(componentEntry.name, "");
                        m_componentDatabase[key] = pluginEntry.handle.filename();
                    }
                }
            }
            m_pluginFiles[pluginEntry.plugin->getName()] = filepath;
            closePlugin(pluginEntry);
        } catch (std::exception& e) {
            std::cerr << "PluginManager::processDirectory(\"" << directory
                      << "\"): error:" << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }
}

void PluginManager::refreshPluginInfo()
{
    for (std::size_t i = 0 ; i < m_pluginDirectories.size() ; i++) {
        try {
            processDirectory(m_pluginDirectories[i]);
        } catch (std::exception& e) {
            std::cerr << "PluginManager::refreshPluginInfo()): error:" << std::endl;
            std::cerr << e.what() << std::endl;
        }
    }
}

bool PluginManager::canFindComponent(std::string componentName,
                                     std::string templateParameters)
{
    const ComponentID key(componentName, templateParameters);
    return m_componentDatabase.find(key) != m_componentDatabase.end();
}

Plugin& PluginManager::loadPluginContaining(std::string componentName,
                                            std::string templateParameters)
{
    const ComponentID key(componentName, templateParameters);
    const ComponentDatabase::iterator i = m_componentDatabase.find(key);
    if (i == m_componentDatabase.end())
        throw std::runtime_error("PluginManager: no known plugin contains"
                                 " the '" + componentName + "' component"
                                 " with the following template parameters: "
                                 "'" + templateParameters + "'");
    return loadPlugin(i->second);
}

const sofa::core::Plugin& PluginManager::getLoadedPlugin(const std::string& pluginName) const {
    if (m_loadedPlugins.find(pluginName) != m_loadedPlugins.end())
        return *m_loadedPlugins.find(pluginName)->second.plugin;
    else if (m_loadedLegacyPlugins.find(pluginName) != m_loadedLegacyPlugins.end())
        return *m_loadedLegacyPlugins.find(pluginName)->second.plugin;
    else
        throw std::runtime_error("PluginManager::getLoadedPlugin(): plugin '" + pluginName+ "' not found.");
}

const std::map<std::string, PluginManager::LoadedPlugin>& PluginManager::getLoadedPlugins() const {
    return m_loadedPlugins;
}

const std::map<std::string, PluginManager::LoadedPlugin>& PluginManager::getLoadedLegacyPlugins() const {
    return m_loadedLegacyPlugins;
}

const std::vector<std::string>& PluginManager::getPluginDirectories() const {
    return m_pluginDirectories;
}

const PluginManager::ComponentDatabase& PluginManager::getComponentDatabase() const {
    return m_componentDatabase;
}

}

}

