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
#include <SceneChecking/SceneCheckDeprecatedComponents.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>
using sofa::simulation::Node;

#include <sofa/helper/ComponentChange.h>
using sofa::helper::lifecycle::deprecatedComponents;

namespace sofa::_scenechecking_
{

const bool SceneCheckDeprecatedComponentsRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckDeprecatedComponents::newSPtr());

const std::string SceneCheckDeprecatedComponents::getName()
{
    return "SceneCheckDeprecatedComponents";
}

const std::string SceneCheckDeprecatedComponents::getDesc()
{
    return "Check there is not deprecated components in the scenegraph";
}

void SceneCheckDeprecatedComponents::doInit(Node* node)
{
    SOFA_UNUSED(node);
}

void SceneCheckDeprecatedComponents::doCheckOn(Node* node)
{
    if (node == nullptr)
        return;

    for (auto& object : node->object )
    {
        if (const sofa::core::Base* o = object.get())
        {
            if( deprecatedComponents.find( o->getClassName() ) != deprecatedComponents.end() )
            {
                msg_deprecated(o) << this->getName() << ": "
                    << deprecatedComponents.at(o->getClassName()).getMessage();
            }
        }
    }
}

void SceneCheckDeprecatedComponents::doPrintSummary()
{}

std::shared_ptr<SceneCheckDeprecatedComponents> SceneCheckDeprecatedComponents::newSPtr()
{
    return std::shared_ptr<SceneCheckDeprecatedComponents>(new SceneCheckDeprecatedComponents());
}

} //namespace sofa::_scenechecking_
