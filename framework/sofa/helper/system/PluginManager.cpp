#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <fstream>
namespace sofa
{
namespace helper
{
namespace system
{

namespace
{
#ifdef NDEBUG
const std::string pluginsIniFile = "config/sofaplugins_release.ini";
#else
const std::string pluginsIniFile = "config/sofaplugins_debug.ini";
#endif

template <class LibraryEntry>
bool getPluginEntry(LibraryEntry& entry, DynamicLibrary* plugin, std::ostream* errlog)
{
    typedef typename LibraryEntry::FuncPtr FuncPtr;
    entry.func = (FuncPtr)plugin->getSymbol(entry.symbol,errlog);
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

PluginManager::~PluginManager()
{

    writeToIniFile();
}

void PluginManager::readFromIniFile()
{
    std::string path= pluginsIniFile;
    if ( !DataRepository.findFile(path) )
    {
        path = DataRepository.getFirstPath() + "/" + pluginsIniFile;
        std::ofstream ofile(path.c_str());
        ofile << "";
        ofile.close();
    }
    else path = DataRepository.getFile( pluginsIniFile );

    std::ifstream instream(path.c_str());
    std::string pluginPath;

    while(std::getline(instream,pluginPath))
    {
        if(loadPlugin(pluginPath))
            m_pluginMap[pluginPath].initExternalModule();
    }
    instream.close();
}

void PluginManager::writeToIniFile()
{
    std::string path= pluginsIniFile;
    if ( !DataRepository.findFile(path) )
    {
        path = DataRepository.getFirstPath() + "/" + pluginsIniFile;
        std::ofstream ofile(path.c_str(),std::ios::out);
        ofile << "";
        ofile.close();
    }
    else path = DataRepository.getFile( pluginsIniFile );
    std::ofstream outstream(path.c_str());
    PluginIterator iter;
    for( iter = m_pluginMap.begin(); iter!=m_pluginMap.end(); ++iter)
    {
        const std::string& pluginPath = (iter->first);
        outstream << pluginPath << "\n";
    }
    outstream.close();
}

bool PluginManager::loadPlugin(const std::string& path, std::ostream* errlog)
{
    std::string pluginPath(path);
    if( !PluginRepository.findFile(pluginPath,"",errlog) )
    {
        return false;
    }
    if(m_pluginMap.find(pluginPath) != m_pluginMap.end() )
    {
        (*errlog) << "Plugin " << pluginPath << " already in PluginManager" << std::endl;
        return false;
    }
    DynamicLibrary* d  = DynamicLibrary::load(pluginPath, errlog);
    Plugin p;
    if( d == NULL )
    {
        return false;
    }
    else
    {
        if(! getPluginEntry(p.initExternalModule,d,errlog) ) return false;
        getPluginEntry(p.getModuleName,d,errlog);
        getPluginEntry(p.getModuleDescription,d,errlog);
        getPluginEntry(p.getModuleLicense,d,errlog);
        getPluginEntry(p.getModuleComponentList,d,errlog);
        getPluginEntry(p.getModuleVersion,d,errlog);
    }

    p.dynamicLibrary = boost::shared_ptr<DynamicLibrary>(d);
    m_pluginMap[pluginPath] = p;

    return true;
}

bool PluginManager::unloadPlugin(const std::string &path, std::ostream *errlog)
{
    PluginMap::iterator iter;
    iter = m_pluginMap.find(path);
    if( iter == m_pluginMap.end() )
    {
        (*errlog) << "Plugin " << path << "not in PluginManager" << std::endl;
        return false;
    }
    else
    {
        m_pluginMap.erase(iter);
        return true;
    }
}

void PluginManager::initRecentlyOpened()
{
    readFromIniFile();
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



}


}

}


