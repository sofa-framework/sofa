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
#include <memory>
#include <optional>
#include <sofa/simulation/config.h>
#include <sofa/simulation/mappinggraph/MappingGraphVisitor.h>

namespace sofa::simulation
{


/**
 * @brief Abstract base class for nodes in the mapping graph.
 *
 * Provides common functionality for tracking prerequisites (parents) and dependencies
 * (children), forming a Directed Acyclic Graph (DAG). It implements the Visitor pattern.
 */
class SOFA_SIMULATION_CORE_API BaseMappingGraphNode : public std::enable_shared_from_this<BaseMappingGraphNode>
{
public:
    using SPtr = std::shared_ptr<BaseMappingGraphNode>;
    friend class MappingGraph;

    virtual ~BaseMappingGraphNode() = default;

    /**
     * @brief Accepts a visitor, allowing the graph to be processed by an external algorithm.
     * @param visitor The concrete visitor implementation.
     */
    virtual void accept(MappingGraphVisitor& visitor) const = 0;

    /**
     * @brief Gets the name of the component represented by this node.
     * @return The name string.
     */
    virtual std::string getName() const { return {}; }

    enum class NodeType {
        MechanicalState,
        Mapping,
        Component,
        Group
    };

    virtual NodeType getType() const = 0;

    /**
     * @return True if the node has an ancestor which is a mapping node
     */
    bool isMapped() const;

private:
    sofa::type::vector<SPtr> m_parents;   ///< prerequisite nodes (nodes pointing to this one)
    sofa::type::vector<SPtr> m_children;  ///< dependent nodes (nodes pointed from this one)

    // Mutable counter used during traversal (reset before each traversal).
    mutable int m_pendingCount = 0;

    mutable std::optional<bool> m_isMapped;
};


}
