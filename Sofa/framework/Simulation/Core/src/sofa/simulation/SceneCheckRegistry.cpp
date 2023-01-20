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
#include <sofa/simulation/SceneCheckRegistry.h>
#include <algorithm>

namespace sofa::simulation
{

bool SceneCheckRegistry::addToRegistry(const SceneCheck::SPtr& sceneCheck)
{
    const auto it = std::find(m_registeredSceneChecks.begin(), m_registeredSceneChecks.end(), sceneCheck);
    const auto found = it != m_registeredSceneChecks.end();
    if(!found)
    {
        m_registeredSceneChecks.push_back(sceneCheck);
    }
    return !found;
}

void SceneCheckRegistry::removeFromRegistry(const SceneCheck::SPtr& sceneCheck)
{
    m_registeredSceneChecks.erase( std::remove( m_registeredSceneChecks.begin(), m_registeredSceneChecks.end(), sceneCheck ), m_registeredSceneChecks.end() );
}

void SceneCheckRegistry::clearRegistry()
{
    m_registeredSceneChecks.clear();
}

const type::vector<SceneCheck::SPtr>& SceneCheckRegistry::getRegisteredSceneChecks() const
{
    return m_registeredSceneChecks;
}

}
