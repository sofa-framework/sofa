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
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Utils.h>
#include <sofa/helper/logging/Messaging.h>
#include <fstream>
#include <sofa/helper/system/config.h>

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
    m_searchPaths = PluginRepository.getPaths();
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
    std::string pluginPath;

    while(std::getline(instream,pluginPath))
    {
        if(loadPlugin(pluginPath))
            m_pluginMap[pluginPath].initExternalModule();
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
        outstream << pluginPath << "\n";
    }
    outstream.close();
}

bool PluginManager::loadPluginByPath(const std::string& pluginPath, std::ostream* errlog)
{
    if (pluginIsLoaded(pluginPath))
    {
        const std::string msg = "Plugin already loaded: " + pluginPath;
//        Logger::getMainLogger().log(Logger::Warning, msg, "PluginManager");
        if (errlog) (*errlog) << msg << std::endl;
        return false;
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
        getPluginEntry(p.getModuleDescription,d);
        getPluginEntry(p.getModuleLicense,d);
        getPluginEntry(p.getModuleComponentList,d);
        getPluginEntry(p.getModuleVersion,d);
    }

    p.dynamicLibrary = d;
    m_pluginMap[pluginPath] = p;
    p.initExternalModule();

    msg_info("PluginManager") << "Loaded plugin: " << pluginPath;
    return true;
}

bool PluginManager::loadPluginByName(const std::string& pluginName, std::ostream* errlog)
{
    std::string pluginPath = findPlugin(pluginName);

    if (pluginPath != "")
    {
        return loadPluginByPath(pluginPath, errlog);
    }
    else
    {
        const std::string msg = "Plugin not found: \"" + pluginName + "\"";
        if (errlog) (*errlog) << msg << std::endl;
        else msg_error("PluginManager") << msg;

        return false;
    }
}

bool PluginManager::loadPlugin(const std::string& plugin, std::ostream* errlog)
{
    // If 'plugin' ends with ".so", ".dll" or ".dylib", this is a path
    const std::string dotExt = "." + DynamicLibrary::extension;
    if (std::equal(dotExt.rbegin(), dotExt.rend(), plugin.rbegin()))
        return loadPluginByPath(plugin, errlog);
    else
        return loadPluginByName(plugin, errlog);
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
        return true;
    }
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



std::string PluginManager::findPlugin(const std::string& pluginName, bool ignoreCase)
{
    std::string name(pluginName);
#ifdef SOFA_LIBSUFFIX
    name += sofa_tostring(SOFA_LIBSUFFIX);
#endif
#if defined(_DEBUG) && defined(_MSC_VER) 
	 name += "_d";
#endif
    const std::string libName = DynamicLibrary::prefix + name + "." + DynamicLibrary::extension;

    // First try: case sensitive
    for (std::vector<std::string>::iterator i = m_searchPaths.begin(); i!=m_searchPaths.end(); i++)
    {
        const std::string path = *i + "/" + libName;
        if (FileSystem::exists(path))
            return path;
    }
    // Second try: case insensitive
    if (ignoreCase)
    {
        for (std::vector<std::string>::iterator i = m_searchPaths.begin(); i!=m_searchPaths.end(); i++)
        {
            const std::string& dir = *i;
            const std::string path = dir + "/" + libName;
            const std::string downcaseLibName = Utils::downcaseString(libName);
            std::vector<std::string> files;
            FileSystem::listDirectory(dir, files);
            for(std::vector<std::string>::iterator j = files.begin(); j != files.end(); j++)
            {
                const std::string& filename = *j;
                const std::string downcaseFilename = Utils::downcaseString(filename);
                if (downcaseFilename == downcaseLibName)
                    return dir + "/" + filename;
            }
        }
    }
    return "";
}

bool PluginManager::pluginIsLoaded(const std::string& pluginPath)
{
    return m_pluginMap.find(pluginPath) != m_pluginMap.end();
}

}

}

}
