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
#include "SceneCheckDuplicatedName.h"

#include <sofa/simulation/Node.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>

namespace sofa::_scenechecking_
{

const bool SceneCheckDuplicatedNameRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckDuplicatedName::newSPtr());

using sofa::simulation::Node;

const std::string SceneCheckDuplicatedName::getName()
{
    return "SceneCheckDuplicatedName";
}

const std::string SceneCheckDuplicatedName::getDesc()
{
    return "Check there is not duplicated name in the scenegraph";
}

void SceneCheckDuplicatedName::doInit(sofa::simulation::Node* node)
{
    SOFA_UNUSED(node);
    m_hasDuplicates = false;
    m_duplicatedMsg.str("");
    m_duplicatedMsg.clear();
}

void SceneCheckDuplicatedName::doCheckOn(sofa::simulation::Node* node)
{
    std::map<std::string, int> duplicated;
    for (auto& object : node->object )
    {
        if( duplicated.find(object->getName()) == duplicated.end() )
            duplicated[object->getName()] = 0;
        duplicated[object->getName()]++;
    }

    for (auto& child : node->child )
    {
        if( duplicated.find(child->getName()) == duplicated.end() )
            duplicated[child->getName()] = 0;
        duplicated[child->getName()]++;
    }

    std::stringstream tmp;
    for(auto& p : duplicated)
    {
        if(p.second!=1)
        {
            tmp << "'" << p.first << "', ";
        }
    }

    if(!tmp.str().empty())
    {
        m_hasDuplicates = true;
        m_duplicatedMsg << "- Found duplicated names [" << tmp.str() << "] in node '"<<  node->getPathName() << "'" << msgendl;
    }
}

void SceneCheckDuplicatedName::doPrintSummary()
{
    if(m_hasDuplicates)
    {
        msg_warning(this->getName()) << msgendl
                                     << m_duplicatedMsg.str()
                                     << "Nodes with similar names at the same level in your scene can "
                                        "crash certain operations, please rename them";
    }
}


} // namespace sofa::_scenechecking_
