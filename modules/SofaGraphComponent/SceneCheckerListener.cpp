/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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

#include <SofaGraphComponent/SceneCheckAPIChange.h>
using sofa::simulation::scenechecking::SceneCheckAPIChange;
#include <SofaGraphComponent/SceneCheckMissingRequiredPlugin.h>
using sofa::simulation::scenechecking::SceneCheckMissingRequiredPlugin;
#include <SofaGraphComponent/SceneCheckDuplicatedName.h>
using sofa::simulation::scenechecking::SceneCheckDuplicatedName;
#include <SofaGraphComponent/SceneCheckUsingAlias.h>
using sofa::simulation::scenechecking::SceneCheckUsingAlias;

namespace sofa
{
namespace simulation
{
namespace _scenechecking_
{


SceneCheckerListener::SceneCheckerListener()
{
    m_sceneChecker.addCheck(SceneCheckAPIChange::newSPtr());
    m_sceneChecker.addCheck(SceneCheckDuplicatedName::newSPtr());
    m_sceneChecker.addCheck(SceneCheckMissingRequiredPlugin::newSPtr());
    m_sceneChecker.addCheck(SceneCheckUsingAlias::newSPtr());
}

SceneCheckerListener* SceneCheckerListener::getInstance()
{
    static SceneCheckerListener sceneLoaderListener;
    return &sceneLoaderListener;
}

void SceneCheckerListener::rightAfterLoadingScene(sofa::simulation::Node::SPtr node)
{
    m_sceneChecker.validate(node.get());
}


} // namespace _scenechecking_
} // namespace simulation
} // namespace sofa
