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
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileSystem.h>
#include <sofa/helper/Logger.h>
#include <fstream>

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

bool PluginManager::loadPlugin(const std::string& pluginPath, std::ostream* errlog)
{
    if (pluginIsLoaded(pluginPath))
    {
        const std::string msg = "Plugin already loaded: " + pluginPath;
        Logger::getMainLogger().log(Logger::Warning, msg, "PluginManager::load()");
        if (errlog) (*errlog) << msg << std::endl;
        return false;
    }

    if (!FileSystem::exists(pluginPath))
    {
        const std::string msg = "File not found: " + pluginPath;
        Logger::getMainLogger().log(Logger::Error, msg, "PluginManager::load()");
        if (errlog) (*errlog) << msg << std::endl;
        return false;
    }

    DynamicLibrary::Handle d  = DynamicLibrary::load(pluginPath);
    Plugin p;
    if( ! d.isValid() )
    {
        const std::string msg = "Plugin loading failed (" + pluginPath + "): " + DynamicLibrary::getLastError();
        Logger::getMainLogger().log(Logger::Error, msg, "PluginManager::load()");
        if (errlog) (*errlog) << msg << std::endl;
        return false;
    }
    else
    {
        if(! getPluginEntry(p.initExternalModule,d))
        {
            const std::string msg = "Plugin loading failed (" + pluginPath + "): function initExternalModule() not found";
            Logger::getMainLogger().log(Logger::Error, msg, "PluginManager::load()");
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

    Logger::getMainLogger().log(Logger::Info, "Loaded plugin: " + pluginPath, "PluginManager");
    return true;
}

bool PluginManager::unloadPlugin(const std::string &pluginPath, std::ostream* errlog)
{
    if(!pluginIsLoaded(pluginPath))
    {
        const std::string msg = "Plugin not loaded: " + pluginPath;
        Logger::getMainLogger().log(Logger::Error, msg, "PluginManager::unloadPlugin()");
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

std::string PluginManager::findPlugin(const std::string& pluginName)
{
    std::string name(pluginName);
#ifdef SOFA_LIBSUFFIX
    name += sofa_tostring(SOFA_LIBSUFFIX);
#endif
    const std::string libName = DynamicLibrary::prefix + name + "." + DynamicLibrary::extension;

    for(std::vector<std::string>::iterator i = m_searchPaths.begin(); i!=m_searchPaths.end(); i++)
    {
        const std::string path = *i + "/" + libName;
        if (FileSystem::exists(path))
            return path;
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
