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

enum class SOFA_SIMULATION_CORE_API MappingGraphDirection : bool
{
    FORWARD,
    BACKWARD
};

/**
 * Find independent mappings, i.e. mappings without dependencies.
 */
SOFA_SIMULATION_CORE_API
void findIndependentMappings(const std::vector<sofa::core::BaseMapping*>& allMappings,
                             std::queue<sofa::core::BaseMapping*>& independentMappings,
                             MappingGraphDirection direction = MappingGraphDirection::FORWARD);

/**
 * Invoke a callable on the mappings in the given context, according to a topological order of the
 * mappings such that calls to mappings are made according to the dependencies of the mapping graph.
 */
template <class Callable>
void mappingGraphBreadthFirstTraversal(
    sofa::core::objectmodel::BaseContext* context,
    Callable f,
    bool filterNonMechanicalMappings = true,
    MappingGraphDirection direction = MappingGraphDirection::FORWARD)
requires std::is_invocable_v<Callable, sofa::core::BaseMapping*>
{
    assert(context);

    auto mappingList =
    context->getObjects<sofa::core::BaseMapping>(sofa::core::objectmodel::BaseContext::SearchDirection::SearchDown);

    if (filterNonMechanicalMappings)
    {
        std::erase_if(mappingList, [](const sofa::core::BaseMapping* mapping){return !mapping->isMechanical();});
    }

    std::queue<sofa::core::BaseMapping*> mappingsToProcess;

    while (!mappingList.empty())
    {
        /// Among the current mapping list, find the mappings that are not the input of other mappings
        findIndependentMappings(mappingList, mappingsToProcess, direction);

        if (mappingsToProcess.empty())
        {
            msg_error("MappingGraph") << "Cannot find next mappings to process. Abort.";
            break;
        }

        /// The callable can be invoked safely on the mappings in the queue because they don't
        /// depend on other mappings
        while (!mappingsToProcess.empty())
        {
            auto* mapping = mappingsToProcess.front();
            mappingsToProcess.pop();

            std::invoke(f, mapping);

            std::erase(mappingList, mapping);
        }

    }
}

}
