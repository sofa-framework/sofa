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
#include "SceneCheckAPIChange.h"

#include <string>

#include <sofa/version.h>
#include <sofa/core/objectmodel/Base.h>
using sofa::core::objectmodel::Base;

#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <sofa/component/sceneutility/APIVersion.h>
using sofa::component::sceneutility::APIVersion;

#include <sofa/simulation/SceneCheckMainRegistry.h>

namespace sofa::_scenechecking_
{

// const bool SceneCheckAPIChangeRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckAPIChange::newSPtr());

SceneCheckAPIChange::SceneCheckAPIChange()
{
    installDefaultChangeSets();
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

void SceneCheckAPIChange::doInit(sofa::simulation::Node* node)
{
    std::stringstream version;
    version << SOFA_VERSION / 10000 << "." << SOFA_VERSION / 100 % 100;
    m_currentApiLevel = version.str();

    APIVersion* apiversion {nullptr};
    /// 1. Find if there is an APIVersion component in the scene. If there is none, warn the user and set
    /// the version to 17.06 (the last version before it was introduced). If there is one...use
    /// this component to request the API version requested by the scene.
    node->getTreeObject(apiversion);
    if(!apiversion)
    {
        msg_info(this->getName()) << "No 'APIVersion' component in scene. Using the default APIVersion level: " << m_selectedApiLevel;
    }
    else
    {
        m_selectedApiLevel = apiversion->getApiLevel();
    }
}

void SceneCheckAPIChange::doPrintSummary()
{
}

void SceneCheckAPIChange::doCheckOn(sofa::simulation::Node* node)
{
    if(node==nullptr)
        return;

    for (auto& object : node->object )
    {
        Base* o = object.get();

        if(m_selectedApiLevel != m_currentApiLevel && m_changesets.find(m_selectedApiLevel) != m_changesets.end())
        {
            for(auto& hook : m_changesets[m_selectedApiLevel])
            {
                hook(o);
            }
        }
    }
}

void SceneCheckAPIChange::installDefaultChangeSets()
{
    // Template of addHookInChangeSet
    // addHookInChangeSet warns the user about changes that occured within a component
    // (change in API, behavior, default values, etc.)
    /*
    addHookInChangeSet("17.06", [this](Base* o){
        if(o->getClassName() == "BoxStiffSpringForceField" )
            msg_warning(o) << this->getName() << ": "
                           << "BoxStiffSpringForceField have changed since 17.06. To use the old behavior you need to set parameter 'forceOldBehavior=true'";
    });
    */
}

void SceneCheckAPIChange::addHookInChangeSet(const std::string& version, ChangeSetHookFunction fct)
{
    m_changesets[version].push_back(fct);
}

} // namespace sofa::_scenechecking_
