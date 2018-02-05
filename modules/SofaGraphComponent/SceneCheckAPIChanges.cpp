/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#include <sofa/version.h>
#include <string>
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base ;

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/Visitor.h>
#include <sofa/helper/system/PluginManager.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/helper/deprecatedcomponents.h>
using sofa::helper::deprecatedcomponents::components ;
using sofa::helper::deprecatedcomponents::messages ;
using sofa::helper::deprecatedcomponents::indexName ;
using sofa::helper::deprecatedcomponents::indexMessage ;



#include "SceneCheckAPIChanges.h"
#include "RequiredPlugin.h"

#include "APIVersion.h"
using sofa::component::APIVersion ;

namespace sofa
{
namespace simulation
{
namespace _scenecheckapichange_
{

SceneCheckAPIChange::SceneCheckAPIChange()
{
    installDefaultChangeSets() ;
}

SceneCheckAPIChange::~SceneCheckAPIChange()
{

}

const std::string SceneCheckAPIChange::getName()
{
    return "SceneCheckAPIChange";
}

const std::string SceneCheckAPIChange::getDesc()
{
    return "Check for each component that the behavior have not changed since reference version of sofa.";
}

void SceneCheckAPIChange::doInit(Node* node)
{
    std::stringstream version;
    version << SOFA_VERSION / 10000 << "." << SOFA_VERSION / 100 % 100;
    m_currentApiLevel = version.str();

    APIVersion* apiversion {nullptr} ;
    /// 1. Find if there is an APIVersion component in the scene. If there is none, warn the user and set
    /// the version to 17.06 (the last version before it was introduced). If there is one...use
    /// this component to request the API version requested by the scene.
    node->getTreeObject(apiversion) ;
    if(!apiversion)
    {
        msg_info("SceneChecker") << "The 'APIVersion' directive is missing in the current scene. Switching to the default APIVersion level '"<< m_selectedApiLevel <<"' " ;
    }
    else
    {
        m_selectedApiLevel = apiversion->getApiLevel() ;
    }
}

void SceneCheckAPIChange::doCheckOn(Node* node)
{
    if(node==nullptr)
        return ;

    for (auto& object : node->object )
    {
        if(m_selectedApiLevel != m_currentApiLevel && m_changesets.find(m_selectedApiLevel) != m_changesets.end())
        {
            for(auto& hook : m_changesets[m_selectedApiLevel])
            {
                hook(object.get());
            }
        }

    }
}


void SceneCheckAPIChange::installDefaultChangeSets()
{
    addHookInChangeSet("17.06", [](Base* o){
        if(o->getClassName() == "BoxStiffSpringForceField" )
            msg_warning(o) << "BoxStiffSpringForceField have changed since 17.06. To use the old behavior you need to set parameter 'forceOldBehavior=true'" ;
    }) ;

    addHookInChangeSet("17.06", [](Base* o){
        if( components.find( o->getClassName() ) != components.end() )
        {
            auto& msg = components[o->getClassName()] ;
            std::string str = msg[indexName];

            /// Replace the string by the default one.
            if( messages.find( str ) != messages.end() ){
                str = messages[str] ;
            }

            if(msg.size() >= indexMessage )
            {
                msg_warning(o) << o->getClassName()
                                   << str ;
            }
            else {
                msg_warning(o) << o->getClassName()
                               << str
                               << msg[indexMessage] ;

            }
        }
    }) ;
}

void SceneCheckAPIChange::addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct)
{
    m_changesets[version].push_back(fct) ;
}


} // _scenecheckapichange_

} // namespace simulation

} // namespace sofa

