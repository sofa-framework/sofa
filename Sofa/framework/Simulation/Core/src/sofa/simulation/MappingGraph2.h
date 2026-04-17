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

#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/BaseForceField.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/BaseMechanicalState.h>
#include <sofa/simulation/Node.h>

#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace sofa::simulation
{
class TaskScheduler;

// ---------------------------------------------------------------------------
// Visitor interface
// ---------------------------------------------------------------------------
class MappingGraphVisitor
{
public:
    virtual ~MappingGraphVisitor() = default;

    virtual void visit(core::behavior::BaseMechanicalState&) {}
    virtual void visit(core::BaseMapping&) {}
    virtual void visit(core::behavior::BaseForceField&) {}
    virtual void visit(core::behavior::BaseMass&) {}
};

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------
class BaseMappingGraphNode : public std::enable_shared_from_this<BaseMappingGraphNode>
{
public:
    using SPtr = std::shared_ptr<BaseMappingGraphNode>;
    friend class MappingGraph2;
    virtual ~BaseMappingGraphNode() = default;

    virtual void accept(MappingGraphVisitor& visitor) const = 0;

    virtual std::string getName() const { return {}; }

private:
    sofa::type::vector<SPtr> m_parents;   // prerequisite nodes
    sofa::type::vector<SPtr> m_children;  // dependent nodes

    // Mutable counter used during traversal (reset before each traversal).
    mutable int m_pendingCount = 0;
};

template<class TComponent>
class MappingGraphNode : public BaseMappingGraphNode
{
public:
    using SPtr = std::shared_ptr<MappingGraphNode>;
    friend class MappingGraph2;
    explicit MappingGraphNode(typename TComponent::SPtr s)
        : m_component(std::move(s))
    {}

    void accept(MappingGraphVisitor& visitor) const override
    {
        if (m_component)
        {
            visitor.visit(*m_component);
        }
    }

    std::string getName() const override
    {
        return m_component->getName();
    }

private:
    typename TComponent::SPtr m_component;
};

class ComponentGroupMappingGraphNode : public BaseMappingGraphNode
{
public:
    using SPtr = std::shared_ptr<ComponentGroupMappingGraphNode>;
    void accept(MappingGraphVisitor& visitor) const override { SOFA_UNUSED(visitor); }

    std::string getName() const override
    {
        return "group";
    }
};

// ---------------------------------------------------------------------------
// Mapping graph
// ---------------------------------------------------------------------------
class SOFA_SIMULATION_CORE_API MappingGraph2
{
public:
    using MappingInputs = type::vector<core::behavior::BaseMechanicalState*>;

    // ------------------------------------------------------------------
    // Input: flat lists collected from the scene DAG (or any other source).
    // Add further leaf component lists as needed.
    // ------------------------------------------------------------------
    struct SOFA_SIMULATION_CORE_API InputLists
    {
        sofa::type::vector<core::behavior::BaseMechanicalState*> mechanicalStates;
        sofa::type::vector<core::BaseMapping*> mappings;
        sofa::type::vector<core::behavior::BaseForceField*> forceFields;
        sofa::type::vector<core::behavior::BaseMass*> masses;

        static InputLists makeFromNode(core::objectmodel::BaseContext* node);
        static InputLists makeFromNode(core::objectmodel::BaseContext::SPtr node) { return makeFromNode(node.get()); }
    };

    MappingGraph2() = default;
    explicit MappingGraph2(const InputLists& input);
    explicit MappingGraph2(core::objectmodel::BaseContext* node);

    /// Return the node used to start the exploration of the scene graph in order to build the mapping graph
    [[nodiscard]] core::objectmodel::BaseContext* getRootNode() const;

    /// Return the list of all mechanical states that are not mapped
    [[nodiscard]] const sofa::type::vector<core::behavior::BaseMechanicalState*>& getMainMechanicalStates() const;

    /// Return the list of mechanical states which are:
    /// 1) non-mapped
    /// 2) input of a mapping involving the provided mechanical state as an output.
    /// The search is recursive (more than one level of mapping).
    MappingInputs getTopMostMechanicalStates(core::behavior::BaseMechanicalState* state) const;

    /// Return the list of mechanical states which are:
    /// 1) non-mapped
    /// 2) input of a mapping involving the mechanical states associated to the provided force field as an output.
    /// The search is recursive (more than one level of mapping)
    template<class TComponent> requires !std::derived_from<TComponent, core::behavior::BaseMechanicalState>
    MappingInputs getTopMostMechanicalStates(TComponent* component) const;

    [[nodiscard]] bool hasAnyMapping() const;

    /// Return the sum of the degrees of freedom of all main mechanical states
    [[nodiscard]] sofa::Size getTotalNbMainDofs() const;

    /// Return where in the global matrix the provided mechanical state writes its contribution
    type::Vec2u getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* mstate) const;
    /// Return where in the global matrix the provided mechanical states writes its contribution
    type::Vec2u getPositionInGlobalMatrix(core::behavior::BaseMechanicalState* a,
                                          core::behavior::BaseMechanicalState* b) const;

    // ------------------------------------------------------------------
    // Top-down traversal: roots (unmapped states) → leaves (components).
    //
    // Invariants:
    //   • A BaseMechanicalState is visited only after every BaseMapping
    //     that produces it as output has been visited.
    //   • A BaseMapping is visited only after every input
    //     BaseMechanicalState has been visited.
    //   • A leaf component is visited only after every BaseMechanicalState
    //     connected to it has been visited.
    //   • Each node is visited exactly once.
    // ------------------------------------------------------------------
    void traverseTopDown(MappingGraphVisitor& visitor) const;

    // ------------------------------------------------------------------
    // Bottom-up traversal: leaves → roots.
    //
    // Mirror invariants of top-down, reversed.
    // ------------------------------------------------------------------
    void traverseBottomUp(MappingGraphVisitor& visitor) const;

    void traverseComponentGroups(MappingGraphVisitor& visitor) const;
    void traverseComponentGroups(MappingGraphVisitor& visitor, TaskScheduler* taskScheduler) const;

    [[nodiscard]] bool isBuilt() const;

    // ------------------------------------------------------------------
    // Graph construction
    // ------------------------------------------------------------------
    void build(const InputLists& input);
    void build(core::objectmodel::BaseContext* rootNode);

private:
    /// node used to start the exploration of the scene graph in order to build the mapping graph
    core::objectmodel::BaseContext* m_rootNode { nullptr };

    bool m_isBuilt = false;
    bool m_hasAnyMapping = false;

    sofa::Size m_totalNbMainDofs {};

    /// for each main mechanical states, gives the position of its contribution in the global matrix
    std::map<core::behavior::BaseMechanicalState*, type::Vec2u > m_positionInGlobalMatrix;

    std::vector<BaseMappingGraphNode::SPtr> m_allNodes;
    sofa::type::vector<core::behavior::BaseMechanicalState*> m_roots {};
    std::unordered_map<core::behavior::BaseMechanicalState*, BaseMappingGraphNode*> m_stateIndex;
    std::vector<std::pair<
        std::vector<core::behavior::BaseMechanicalState::SPtr>,
        ComponentGroupMappingGraphNode::SPtr>> m_groupIndex;

    // ------------------------------------------------------------------
    // Kahn-style BFS used by both traversal directions.
    //   topDown == true  → follow children, decrement their pendingCount.
    //   topDown == false → follow parents,  decrement their pendingCount.
    // ------------------------------------------------------------------
    static void processQueue(std::queue<BaseMappingGraphNode*>& ready,
                             MappingGraphVisitor&       visitor,
                             bool                       topDown);

    ComponentGroupMappingGraphNode::SPtr findGroupNode(const std::vector<core::behavior::BaseMechanicalState::SPtr>& states);
    ComponentGroupMappingGraphNode::SPtr findInGroupNodes(const core::behavior::BaseMechanicalState::SPtr state);
    BaseMappingGraphNode* findStateNode(core::behavior::BaseMechanicalState* raw) const;

    // Directed edge from → to.
    // Both parent and child lists hold SPtr so the graph owns all nodes.
    static void addEdge(BaseMappingGraphNode* from, BaseMappingGraphNode* to);
};

template <class TComponent> requires !std::derived_from<TComponent, core::behavior::BaseMechanicalState>
MappingGraph2::MappingInputs MappingGraph2::getTopMostMechanicalStates(TComponent* component) const
{
    if (component == nullptr)
    {
        dmsg_error("MappingGraph") << "Requested mass is invalid";
        return {};
    }

    const auto& associatedMechanicalStates = component->getMechanicalStates();
    MappingInputs topMostMechanicalStates;
    for (auto* mstate : associatedMechanicalStates)
    {
        const auto mstates = getTopMostMechanicalStates(mstate);
        topMostMechanicalStates.insert(topMostMechanicalStates.end(), mstates.begin(), mstates.end());
    }
    return topMostMechanicalStates;
}

}  // namespace sofa::simulation
