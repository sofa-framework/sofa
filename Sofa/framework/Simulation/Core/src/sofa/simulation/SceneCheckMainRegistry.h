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
#pragma once

#include <sofa/simulation/SceneCheckRegistry.h>
#include <mutex>

namespace sofa::simulation
{

class SOFA_SIMULATION_CORE_API SceneCheckMainRegistry
{
public:
    static bool addToRegistry(const SceneCheck::SPtr& sceneCheck);
    static void removeFromRegistry(const SceneCheck::SPtr& sceneCheck);
    static void clearRegistry();
    static const type::vector<SceneCheck::SPtr>& getRegisteredSceneChecks();

private:
    static std::mutex s_mutex;
    static SceneCheckRegistry& getInstance();
};

}
