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
#include <sofa/component/linearsystem/config.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

#include <sofa/simulation/Node.h>

namespace sofa::component::linearsystem
{

using core::behavior::BaseMechanicalState;

/**
 * Connexions betweeen objects through mappings
 *
 * Graph must be built with the build() function.
 */
class SOFA_COMPONENT_LINEARSYSTEM_API MappingGraph
{

public:
    using MappingInputs = type::vector<BaseMechanicalState*>;

    /// Return the node used to start the exploration of the scene graph in order to build the mapping graph
    [[nodiscard]] core::objectmodel::BaseContext* getRootNode() const;
    /// Return the list of all mechanical states which are not mapped
    [[nodiscard]] const sofa::type::vector<BaseMechanicalState*>& getMainMechanicalStates() const;

    /// Return the list of mechanical states which are:
    /// 1) non-mapped
    /// 2) input of a mapping involving the provided mechanical state as an output.
    /// The search is recursive (more than one level of mapping) and is done during mapping graph construction.
    MappingInputs getTopMostMechanicalStates(BaseMechanicalState*) const;

    /// Return the list of mechanical states which are:
    /// 1) non-mapped
    /// 2) input of a mapping involving the mechanical states associated to the provided force field as an output.
    /// The search is recursive (more than one level of mapping) and is done during mapping graph construction.
    MappingInputs getTopMostMechanicalStates(core::behavior::BaseForceField*) const;

    /// Return the list of mechanical states which are:
    /// 1) non-mapped
    /// 2) input of a mapping involving the mechanical states associated to the provided mass as an output.
    /// The search is recursive (more than one level of mapping) and is done during mapping graph construction.
    MappingInputs getTopMostMechanicalStates(core::behavior::BaseMass*) const;

    struct SameGroupComponents
    {
        sofa::type::vector<core::behavior::BaseForceField*> forceFields;
        sofa::type::vector<core::behavior::BaseMass*> masses;
    };

    using ComponentGroups = std::map<BaseMechanicalState*, SameGroupComponents>;

    /// Create groups of components associated to the same mechanical state
    ComponentGroups makeComponentGroups(const sofa::core::ExecParams* params) const;

    [[nodiscard]]
    bool hasAnyMapping() const;

    /// Return true if the provided mechanical state is an output of a mapping
    bool hasAnyMappingInput(BaseMechanicalState*) const;
    /// Return true if the mechanical states associated to the provided force field is an output of a mapping
    bool hasAnyMappingInput(core::behavior::BaseForceField*) const;
    /// Return true if the mechanical states associated to the provided mass is an output of a mapping
    bool hasAnyMappingInput(core::behavior::BaseMass*) const;

    /// Return true if the provided mechanical state has been visited when building the mapping graph
    bool isMechanicalStateInContext(BaseMechanicalState*) const;

    /// Return true if @input is a mapping input of @output. Multiple intermediate mappings are supported
    /// In term of graph connectivity, return true if the two nodes of the directed graph are connected
    bool isMappingInput(BaseMechanicalState* mappingInput, BaseMechanicalState* mappingOutput) const;

    /// Returns all mechanical states which are input of a mapping where the mechanical state in
    /// parameter is an output
    MappingInputs getMappingInputs(BaseMechanicalState*) const;

    sofa::type::vector<core::BaseMapping*> getBottomUpMappingsFrom(BaseMechanicalState*) const;

    /// Return the sum of the degrees of freedom of all main mechanical states
    [[nodiscard]]
    sofa::Size getTotalNbMainDofs() const { return m_totalNbMainDofs; }

    /// Return where in the global matrix the provided mechanical state writes its contribution
    type::Vec2u getPositionInGlobalMatrix(BaseMechanicalState*) const;
    /// Return where in the global matrix the provided mechanical states writes its contribution
    type::Vec2u getPositionInGlobalMatrix(BaseMechanicalState* a, BaseMechanicalState* b) const;

    MappingGraph() = default;

    bool isBuilt() const;

    /// Build the graph: mandatory to get valid data from the functions that use the graph
    void build(const sofa::core::ExecParams* params, core::objectmodel::BaseContext* rootNode);

private:

    /// node used to start the exploration of the scene graph in order to build the mapping graph
    core::objectmodel::BaseContext* m_rootNode { nullptr };

    /// List of all mechanical states in the root context.
    std::vector<BaseMechanicalState*> m_mechanicalStates;

    /// List of all mappings in the root context.
    sofa::type::vector<core::BaseMapping*> m_mappings;

    /// Key: any mechanical state
    /// Value: The list of mapping inputs
    std::map<BaseMechanicalState*, MappingInputs > m_adjacencyList;

    /// List of mechanical states that are non-mapped. They can be involved as a mapping input, but not as an output.
    sofa::type::vector<BaseMechanicalState*> m_mainMechanicalStates;

    /// Association between a mechanical state (the key) and a list of mapping input which are non-mapped. In this list,
    /// the mechanical states are involved as an input, but not as an output. The mechanical state in the key is an
    /// output of mappings (even over multiple levels).
    std::map< BaseMechanicalState*, MappingInputs> m_topMostInputsMechanicalStates;

    /// for each main mechanical states, gives the position of its contribution in the global matrix
    std::map< BaseMechanicalState*, type::Vec2u > m_positionInGlobalMatrix;

    sofa::Size m_totalNbMainDofs {};
    bool m_hasAnyMapping = false;

    void buildAjacencyList();
    void buildMStateRelationships();

};

template<class JacobianMatrixType>
class MappingJacobians
{
    const BaseMechanicalState& m_mappedState;

    std::map< core::behavior::BaseMechanicalState*, std::shared_ptr<JacobianMatrixType> > m_map;

public:

    MappingJacobians() = delete;
    MappingJacobians(const BaseMechanicalState& mappedState) : m_mappedState(mappedState) {}

    void addJacobianToTopMostParent(std::shared_ptr<JacobianMatrixType> jacobian, core::behavior::BaseMechanicalState* topMostParent)
    {
        m_map[topMostParent] = jacobian;
    }

    std::shared_ptr<JacobianMatrixType> getJacobianFrom(core::behavior::BaseMechanicalState* mstate) const
    {
        const auto it = m_map.find(mstate);
        if (it != m_map.end())
            return it->second;
        return nullptr;
    }
};

} //namespace sofa::component::linearsolver
