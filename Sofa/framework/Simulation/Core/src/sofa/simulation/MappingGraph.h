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

#include <sofa/simulation/config.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/objectmodel/BaseContext.h>

#include <queue>
#include <unordered_set>

namespace sofa::simulation
{

enum class SOFA_SIMULATION_CORE_API MappingGraphDirection
{
    TOP_DOWN,
    BOTTOM_UP
};

SOFA_SIMULATION_CORE_API
void findNextMappingsToProcess(const std::vector<sofa::core::BaseMapping*>& mappingList,
                               std::queue<sofa::core::BaseMapping*>& mappings,
                               MappingGraphDirection direction = MappingGraphDirection::TOP_DOWN);

template <class Callable>
void mappingGraphBreadthFirstTraversal(
    sofa::core::objectmodel::BaseContext* context,
    Callable f,
    bool filterNonMechanicalMappings = true,
    MappingGraphDirection direction = MappingGraphDirection::TOP_DOWN)
{
    auto mappingList =
    context->getObjects<sofa::core::BaseMapping>(sofa::core::objectmodel::BaseContext::SearchDirection::SearchDown);

    if (filterNonMechanicalMappings)
    {
        std::erase_if(mappingList, [](const sofa::core::BaseMapping* mapping){return !mapping->isMechanical();});
    }

    std::queue<sofa::core::BaseMapping*> mappingsToProcess;

    while (!mappingList.empty())
    {
        findNextMappingsToProcess(mappingList, mappingsToProcess, direction);

        if (mappingsToProcess.empty())
        {
            msg_warning("MappingGraph") << "Cannot find next mappings to process. Abort.";
            break;
        }

        while (!mappingsToProcess.empty())
        {
            auto* mapping = mappingsToProcess.front();
            mappingsToProcess.pop();

            f(mapping);

            std::erase(mappingList, mapping);
        }

    }
}

}
