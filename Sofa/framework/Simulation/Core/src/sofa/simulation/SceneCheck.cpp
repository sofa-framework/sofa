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
#include <sofa/simulation/SceneCheck.h>

namespace sofa::simulation
{

SceneCheck::~SceneCheck() = default;

void SceneCheck::init(sofa::simulation::Node* node, simulation::SceneLoader* sceneLoader)
{
    SOFA_UNUSED(sceneLoader);
    doInit(node);
}

void SceneCheck::checkOn(sofa::simulation::Node* node, simulation::SceneLoader* sceneLoader)
{
    SOFA_UNUSED(sceneLoader);
    doCheckOn(node);
}

void SceneCheck::printSummary(simulation::SceneLoader* sceneLoader)
{
    SOFA_UNUSED(sceneLoader);
    doPrintSummary();
}

void SceneCheck::doInit(sofa::simulation::Node* node)
{
    SOFA_UNUSED(node);
}

void SceneCheck::doPrintSummary()
{}



}
