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
#include <sofa/component/linearsystem/MappingGraph.h>

#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/core/behavior/BaseProjectiveConstraintSet.h>
#include <sofa/simulation/BaseMechanicalVisitor.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalAccumulateJacobian.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>

namespace sofa::component::linearsystem
{

core::objectmodel::BaseContext* MappingGraph::getRootNode() const
{
    return m_rootNode;
}

const sofa::type::vector<core::behavior::BaseMechanicalState*>& MappingGraph::getMainMechanicalStates() const
{
    return m_mainMechanicalStates;
}

auto MappingGraph::getTopMostMechanicalStates(core::behavior::BaseMechanicalState* mstate) const -> MappingInputs
{
    if (m_rootNode == nullptr)
    {
        msg_error("MappingGraph") << "Graph is not built yet";
    }

    if (mstate == nullptr)
    {
        dmsg_error("MappingGraph") << "Requested mechanical state is invalid";
        return {};
    }

    if (const auto it = m_topMostInputsMechanicalStates.find(mstate); it != m_topMostInputsMechanicalStates.end())
        return it->second;

    return {};
}

auto MappingGraph::getTopMostMechanicalStates(core::behavior::BaseForceField* forceField) const -> MappingInputs
{
    if (forceField == nullptr)
    {
        dmsg_error("MappingGraph") << "Requested force field is invalid";
        return {};
    }

    const auto& associatedMechanicalStates = forceField->getMechanicalStates();
    MappingInputs topMostMechanicalStates;
    for (auto* mstate : associatedMechanicalStates)
    {
        const auto mstates = getTopMostMechanicalStates(mstate);
        topMostMechanicalStates.insert(topMostMechanicalStates.end(), mstates.begin(), mstates.end());
    }
    return topMostMechanicalStates;
}

auto MappingGraph::getTopMostMechanicalStates(core::behavior::BaseMass* mass) const -> MappingInputs
{
    if (mass == nullptr)
    {
        dmsg_error("MappingGraph") << "Requested mass is invalid";
        return {};
    }

    const auto& associatedMechanicalStates = mass->getMechanicalStates();
    MappingInputs topMostMechanicalStates;
    for (auto* mstate : associatedMechanicalStates)
    {
        const auto mstates = getTopMostMechanicalStates(mstate);
        topMostMechanicalStates.insert(topMostMechanicalStates.end(), mstates.begin(), mstates.end());
    }
    return topMostMechanicalStates;
}

class ComponentGroupsVisitor final : public simulation::BaseMechanicalVisitor
{
public:
    ComponentGroupsVisitor(const sofa::core::ExecParams* params, MappingGraph::ComponentGroups& groups)
    : simulation::BaseMechanicalVisitor(params)
    , m_groups(groups)
    {}

    Result fwdMass(simulation::Node*, sofa::core::behavior::BaseMass* mass) override
    {
        if (mass)
        {
            for (auto mstate : mass->getMechanicalStates())
            {
                if (mstate)
                {
                    m_groups[mstate].masses.push_back(mass);
                }
            }
        }
        return Result::RESULT_CONTINUE;
    }
    Result fwdForceField(simulation::Node*, sofa::core::behavior::BaseForceField* ff) override
    {
        if (ff)
        {
            for (auto mstate : ff->getMechanicalStates())
            {
                if (mstate)
                {
                    m_groups[mstate].forceFields.push_back(ff);
                }
            }
        }
        return Result::RESULT_CONTINUE;
    }

private:
    MappingGraph::ComponentGroups& m_groups;
};

MappingGraph::ComponentGroups MappingGraph::makeComponentGroups(const sofa::core::ExecParams* params) const
{
    ComponentGroups groups;
    if (m_rootNode)
    {
        ComponentGroupsVisitor(params, groups).execute(m_rootNode);
    }
    return groups;
}

bool MappingGraph::hasAnyMapping() const
{
    return m_hasAnyMapping;
}

bool MappingGraph::hasAnyMappingInput(core::behavior::BaseMechanicalState* mstate) const
{
    if (m_rootNode == nullptr)
    {
        msg_error("MappingGraph") << "Graph is not built yet";
        return false;
    }

    if (mstate == nullptr)
    {
        msg_error("MappingGraph") << "Requested mechanical state is not valid : cannot get its position in the global matrix";
        return false;
    }

    //only main (non mapped) mechanical states are in this map
    return m_positionInGlobalMatrix.find(mstate) == m_positionInGlobalMatrix.end();
}

bool MappingGraph::hasAnyMappingInput(core::behavior::BaseForceField* forceField) const
{
    for (auto* mstate : forceField->getMechanicalStates())
    {
        if (mstate)
        {
            if (hasAnyMappingInput(mstate))
            {
                return true;
            }
        }
    }
    return false;
}

bool MappingGraph::hasAnyMappingInput(core::behavior::BaseMass* mass) const
{
    for (auto* mstate : mass->getMechanicalStates())
    {
        if (mstate)
        {
            if (hasAnyMappingInput(mstate))
            {
                return true;
            }
        }
    }
    return false;
}

bool MappingGraph::isMechanicalStateInContext(core::behavior::BaseMechanicalState* mstate) const
{
    return std::find(m_mechanicalStates.begin(), m_mechanicalStates.end(), mstate) != m_mechanicalStates.end();
}

bool MappingGraph::isMappingInput(BaseMechanicalState* mappingInput, BaseMechanicalState* mappingOutput) const
{
    const auto it = m_adjacencyList.find(mappingOutput);
    if (it != m_adjacencyList.end())
    {
        const auto& inputs = it->second;
        if (std::find(inputs.begin(), inputs.end(), mappingInput) != inputs.end())
        {
            return true;
        }

        for (auto* directInput : inputs)
        {
            if (isMappingInput(mappingInput, directInput))
            {
                return true;
            }
        }
    }
    return false;
}

sofa::type::vector<BaseMechanicalState*> MappingGraph::getMappingInputs(BaseMechanicalState* mstate) const
{
    const auto it = m_adjacencyList.find(mstate);
    if (it != m_adjacencyList.end())
    {
        return it->second;
    }
    return {};
}

sofa::type::vector<core::BaseMapping*> MappingGraph::getBottomUpMappingsFrom(
    BaseMechanicalState* mstate) const
{
    if (mstate)
    {
        sofa::type::vector<core::BaseMapping*> allMappings;

        sofa::type::vector<core::BaseMapping*> connectedMappings;
        for (auto* mapping : m_mappings)
        {
            if (mapping)
            {
                for (const auto* child : mapping->getMechTo())
                {
                    if (child != nullptr && child == mstate)
                    {
                        connectedMappings.push_back(mapping);
                        break;
                    }
                }
            }
        }

        allMappings.insert(allMappings.end(), connectedMappings.begin(), connectedMappings.end());
        for (auto* mapping : connectedMappings)
        {
            for (auto* parent : mapping->getMechFrom())
            {
                const auto mappings = getBottomUpMappingsFrom(parent);
                allMappings.insert(allMappings.end(), mappings.begin(), mappings.end());
            }
        }

        return allMappings;
    }

    return {};
}

type::Vec2u MappingGraph::getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* mstate) const
{
    if (m_rootNode == nullptr)
    {
        msg_error("MappingGraph") << "Graph is not built yet";
        return type::Vec2u{};
    }

    if (mstate == nullptr)
    {
        msg_error("MappingGraph") << "Requested mechanical state is not valid : cannot get its position in the global matrix";
        return type::Vec2u{};
    }

    if (const auto it = m_positionInGlobalMatrix.find(mstate); it != m_positionInGlobalMatrix.end())
        return it->second;

    msg_error("MappingGraph") << "Requested mechanical state (" << mstate->getPathName() <<
        ") is probably mapped or unknown from the graph: only main mechanical states have an associated submatrix in the global matrix";
    return type::Vec2u{};
}

type::Vec2u MappingGraph::getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* a,
    core::behavior::BaseMechanicalState* b) const
{
    const auto pos_a = getPositionInGlobalMatrix(a);
    const auto pos_b = getPositionInGlobalMatrix(b);
    return {pos_a[0], pos_b[1]};
}

bool MappingGraph::isBuilt() const
{
    return m_rootNode != nullptr;
}

void MappingGraph::build(const sofa::core::ExecParams* params, core::objectmodel::BaseContext* rootNode)
{
    SOFA_UNUSED(params);

    m_rootNode = rootNode;

    m_mainMechanicalStates.clear();
    m_mappings.clear();
    m_adjacencyList.clear();
    m_topMostInputsMechanicalStates.clear();
    m_positionInGlobalMatrix.clear();

    m_totalNbMainDofs = 0;
    m_hasAnyMapping = false;

    if (m_rootNode)
    {
        m_rootNode->getObjects(m_mappings, core::objectmodel::BaseContext::SearchDirection::SearchDown);
    }

    buildAjacencyList();

    m_mechanicalStates.clear();
    if (m_rootNode)
    {
        m_rootNode->getObjects(m_mechanicalStates, core::objectmodel::BaseContext::SearchDirection::SearchDown);
    }

    buildMStateRelationships();
}

void MappingGraph::buildAjacencyList()
{
    for (auto* mapping : m_mappings)
    {
        if (mapping)
        {
            m_hasAnyMapping = true;

            // The mechanical states which are parents of another mechanical state through a mapping are stored in a map for later use
            for (auto* child : mapping->getMechTo())
            {
                if (child != nullptr)
                {
                    for (auto* parent : mapping->getMechFrom())
                    {
                        if (parent != nullptr)
                        {
                            m_adjacencyList[child].push_back(parent);
                        }
                    }
                }
            }
        }
    }
}

void MappingGraph::buildMStateRelationships()
{
    for (auto* mstate : m_mechanicalStates)
    {
        if (mstate == nullptr)
        {
            return;
        }

        auto it = m_adjacencyList.find(mstate);

        if (it == m_adjacencyList.end())
        {
            //mstate has not been found in the map: it's not an output of any mapping
            const auto matrixSize = mstate->getMatrixSize();

            m_mainMechanicalStates.push_back(mstate);
            m_positionInGlobalMatrix[mstate] = sofa::type::Vec2u(m_totalNbMainDofs, m_totalNbMainDofs);

            m_totalNbMainDofs += matrixSize;

            m_topMostInputsMechanicalStates[mstate].push_back(mstate);
        }
        else
        {
            //mstate is the output of at least one mapping and has at least one mechanical state as an input
            MappingGraph::MappingInputs inputs = it->second;
            if (inputs.empty())
            {
                msg_error("MappingGraph") << "Mechanical state " << mstate->getPathName() << " is involved in a mapping, but does not have any valid input mechanical states";
            }
            else
            {
                // continue to check that the input of the mstate is not itself an output of another mstate, and so on
                while(!inputs.empty())
                {
                    auto* visitedMState = inputs.back();
                    inputs.pop_back();
                    it = m_adjacencyList.find(visitedMState);
                    if (it != m_adjacencyList.end())
                    {
                        for (auto* p : it->second)
                            inputs.push_back(p);
                    }
                    else
                    {
                        if (isMechanicalStateInContext(visitedMState))
                        {
                            m_topMostInputsMechanicalStates[mstate].push_back(visitedMState);
                        }
                        else
                        {
                            //the top most input is not in the current context
                        }
                    }
                }
            }

        }
    }
}
}
