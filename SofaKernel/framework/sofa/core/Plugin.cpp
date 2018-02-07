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
