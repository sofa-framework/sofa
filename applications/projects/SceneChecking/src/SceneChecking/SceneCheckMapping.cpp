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
#include <SceneChecking/SceneCheckMapping.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/SceneCheckMainRegistry.h>

#include <algorithm>

namespace sofa::scenechecking
{

const bool SceneCheckMappingRegistered = sofa::simulation::SceneCheckMainRegistry::addToRegistry(SceneCheckMapping::newSPtr());

SceneCheckMapping::~SceneCheckMapping() {}
const std::string SceneCheckMapping::getName() { return "SceneCheckMapping"; }
const std::string SceneCheckMapping::getDesc() { return "Check if the mappings and states inside a Node are consistent regarding the visitor logic."; }

void SceneCheckMapping::doInit(sofa::simulation::Node* node)
{
    m_nodesWithMappingNoState.clear();
    m_nodesWithMappingWrongState.clear();
}

void SceneCheckMapping::doCheckOn(sofa::simulation::Node* node)
{
    if (node->mechanicalMapping != nullptr)
    {
        if (node->mechanicalState || node->state)
        {
            const auto mappingOutput = node->mechanicalMapping->getTo();
            const auto isAnOutput = [&mappingOutput](const sofa::core::BaseState* state) { return std::ranges::find(mappingOutput, state) != mappingOutput.end(); };

            if (node->mechanicalState != nullptr && node->state != nullptr) // two states are present in the Node
            {
                const bool isMechanicalStateAnOutput = isAnOutput(node->mechanicalState);
                const bool isStateAnOutput = isAnOutput(node->state);

                if (!isMechanicalStateAnOutput && !isStateAnOutput)
                {
                    //both mechanicalState and state are not an output of the mapping
                    m_nodesWithMappingWrongState.push_back(node);
                }
                else if (isMechanicalStateAnOutput && !isStateAnOutput)
                {
                    //OK regarding the logic of visitors
                }
                else if (!isMechanicalStateAnOutput && isStateAnOutput)
                {
                    //state is a mapping output, but mechanicalState interferes in the visitor logic
                    m_nodesWithMappingAndTwoStates.push_back(node);
                }
                else
                {
                    // both mechanicalState and state are an output of the mapping: not sure it may happen
                }
            }
            else if (node->mechanicalState != nullptr)
            {
                if (!isAnOutput(node->mechanicalState))
                {
                    // mechanicalState is present in the Node, but it's not an output of the mapping
                    m_nodesWithMappingWrongState.push_back(node);
                }
            }
            else if (node->state != nullptr)
            {
                if (!isAnOutput(node->state))
                {
                    // state is present in the Node, but it's not an output of the mapping
                    m_nodesWithMappingWrongState.push_back(node);
                }
            }
        }
        else
        {
            // no state, no mechanicalState
            m_nodesWithMappingNoState.push_back(node);
        }
    }
}

void SceneCheckMapping::doPrintSummary()
{
    const auto showNodes = [](std::stringstream& ss, sofa::type::vector<sofa::simulation::Node*>& nodes)
    {
        for (auto node : nodes)
        {
            ss << "- " << node->getPathName();
            if (node->mechanicalMapping != nullptr)
            {
                auto mappingOutput = node->mechanicalMapping->getMechTo();
                std::erase_if(mappingOutput, [](const sofa::core::BaseState* state) { return state == nullptr; });
                const auto list = sofa::helper::join(mappingOutput.begin(), mappingOutput.end(),
                    [](const sofa::core::BaseState* state){return state->getPathName();}, ',');

                if (!list.empty())
                {
                    ss << " (it is advised to have " << node->mechanicalMapping->getPathName() << " and its output state(s) [" << list << "] in the same Node";
                }
            }
            ss << msgendl;
        }
    };

    if (!m_nodesWithMappingNoState.empty())
    {
        std::stringstream ss;
        ss << "The following Node(s) contain a mapping but no state: " << msgendl;
        showNodes(ss, m_nodesWithMappingNoState);
        msg_error(getName()) << ss.str();
    }

    if (!m_nodesWithMappingWrongState.empty())
    {
        std::stringstream ss;
        ss << "The following Node(s) contain a mapping and a state, and the state is not an output of the mapping: " << msgendl;
        showNodes(ss, m_nodesWithMappingWrongState);
        msg_error(getName()) << ss.str();
    }

    if (!m_nodesWithMappingWrongState.empty())
    {
        std::stringstream ss;
        ss << "The following Node(s) contain a mapping, a mechanical state and a non-mechanical state. The non-mechanical state is an output of the mapping, but the presence of the mechanical state may interfere and lead to undefined behavior: " << msgendl;
        showNodes(ss, m_nodesWithMappingWrongState);
        msg_error(getName()) << ss.str();
    }
}

}  // namespace sofa::scenechecking
