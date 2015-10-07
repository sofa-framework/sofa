#include <sofa/core/Plugin.h>
#include <cassert>

namespace sofa
{
namespace core
{


const Plugin::ComponentEntry& Plugin::getComponentEntry(std::string name) {
    assert(m_components.find(name) != m_components.end());
    return m_components[name];
}

void Plugin::setDescription(std::string componentName, std::string description) {
    if (m_components.find(componentName) == m_components.end())
        std::cerr << "Plugin::setDescription(): error: no component '" 
                  << componentName << "' in plugin '" << getName() << "'" <<std::endl;
    else
        m_components[componentName].description = description;
}

void Plugin::addAlias(std::string componentName, std::string alias) {
    if (m_components.find(componentName) == m_components.end())
        std::cerr << "Plugin::addAlias(): error: no component '" 
                  << componentName << "' in plugin '" << getName() << "'" <<std::endl;
    else
        m_components[componentName].aliases.insert(alias);
}


}

}
