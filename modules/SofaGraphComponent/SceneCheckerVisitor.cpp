/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#include "SceneCheckerVisitor.h"
#include "RequiredPlugin.h"
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/system/PluginManager.h>

namespace sofa
{
namespace simulation
{
using sofa::component::misc::RequiredPlugin ;
using sofa::core::ObjectFactory ;
using sofa::core::ExecParams ;
using sofa::helper::system::PluginRepository ;
using sofa::helper::system::PluginManager ;

SceneCheckerVisitor::SceneCheckerVisitor(const ExecParams* params) : Visitor(params)
{
}

SceneCheckerVisitor::~SceneCheckerVisitor()
{
}

void SceneCheckerVisitor::validate(Node* node)
{
    helper::vector< RequiredPlugin* > plugins ;
    node->getTreeObjects< RequiredPlugin >(&plugins) ;

    for(auto& plugin : plugins)
        m_requiredPlugins[plugin->getName()] = true ;

    execute(node) ;
}

Visitor::Result SceneCheckerVisitor::processNodeTopDown(Node* node)
{
    for (auto& object : node->object )
    {
        ObjectFactory::ClassEntry entry = ObjectFactory::getInstance()->getEntry(object->getClassName());
        if(!entry.creatorMap.empty())
        {
            ObjectFactory::CreatorMap::iterator it = entry.creatorMap.find(object->getTemplateName());
            if(entry.creatorMap.end() != it && *it->second->getTarget()){
                std::string pluginName = it->second->getTarget() ;
                std::string path = PluginManager::getInstance().findPlugin(pluginName) ;
                if( PluginManager::getInstance().pluginIsLoaded(path)
                    && m_requiredPlugins.find(pluginName) == m_requiredPlugins.end() )
                {
                    msg_warning("SceneChecker") << "This scene is using component '" << object->getClassName() << "'. " << msgendl
                                                << "This component is part of the '" << pluginName << "' plugin but there is no <RequiredPlugin name='" << pluginName << "'> directive in your scene." << msgendl
                                                << "Your scene may not work on a sofa environment that does not have pre-loaded the plugin." << msgendl
                                                << "To fix your scene and remove this warning you need to add the RequiredPlugin directive at the beginning of your scene. ";
                }
            }

        }
    }
    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

