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
#include "SceneCheckerListener.h"

#include <sofa/simulation/Node.h>
#include <SceneChecking/SceneCheckMissingRequiredPlugin.h>
#include <SceneChecking/SceneCheckDuplicatedName.h>
#include <SceneChecking/SceneCheckUsingAlias.h>
#include <SceneChecking/SceneCheckDeprecatedComponents.h>
#include <SceneChecking/SceneCheckCollisionResponse.h>

namespace sofa::_scenechecking_
{

SceneCheckerListener::SceneCheckerListener()
{
    m_sceneChecker.addCheck(SceneCheckDuplicatedName::newSPtr());
    m_sceneChecker.addCheck(SceneCheckMissingRequiredPlugin::newSPtr());
    m_sceneChecker.addCheck(SceneCheckUsingAlias::newSPtr());
    m_sceneChecker.addCheck(SceneCheckDeprecatedComponents::newSPtr());
    m_sceneChecker.addCheck(SceneCheckCollisionResponse::newSPtr());
}

SceneCheckerListener* SceneCheckerListener::getInstance()
{
    static SceneCheckerListener sceneLoaderListener;
    return &sceneLoaderListener;
}

void SceneCheckerListener::rightAfterLoadingScene(sofa::simulation::Node::SPtr node)
{
    if(node.get())
        m_sceneChecker.validate(node.get());
}


} // nnamespace sofa::_scenechecking_
