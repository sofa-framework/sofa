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
#include "Node.h"

#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

namespace sofa::simulation
{

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
    std::vector<SPtr> m_parents;   // prerequisite nodes
    std::vector<SPtr> m_children;  // dependent nodes

    // Mutable counter used during traversal (reset before each traversal).
    mutable int m_pendingCount = 0;
};

template<class TComponent>
class MappingGraphNode : public BaseMappingGraphNode
{
public:
    using SPtr = std::shared_ptr<MappingGraphNode>;
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

template<class TComponent>
MappingGraphNode<TComponent>::SPtr makeMappingGraphNode(typename TComponent::SPtr s)
{
    return MappingGraphNode<TComponent>::SPtr( new MappingGraphNode<TComponent>(s) );
}

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

        static InputLists makeFromNode(sofa::simulation::Node::SPtr node);
    };

    MappingGraph2() = default;
    explicit MappingGraph2(const InputLists& input) { build(input); }

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

    // Accessors for inspection / testing.
    const std::vector<BaseMappingGraphNode::SPtr>& allNodes() const { return m_allNodes; }
    const std::vector<MappingGraphNode<sofa::core::behavior::BaseMechanicalState>*>& roots() const { return m_roots;    }

    [[nodiscard]] bool isBuilt() const;

    // ------------------------------------------------------------------
    // Graph construction
    // ------------------------------------------------------------------
    void build(const InputLists& input);

private:
    bool m_isBuilt = false;
    std::vector<BaseMappingGraphNode::SPtr> m_allNodes;
    std::vector<MappingGraphNode<sofa::core::behavior::BaseMechanicalState>*> m_roots;
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
}
