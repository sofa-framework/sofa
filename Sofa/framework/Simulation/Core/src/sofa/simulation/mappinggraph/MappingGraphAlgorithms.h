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
#include <sofa/simulation/mappinggraph/CallableVisitor.h>
#include <sofa/simulation/mappinggraph/MappingGraphVisitor.h>
#include <sofa/simulation/mappinggraph/VisitorApplication.h>

#include <queue>

namespace sofa::simulation
{

class TaskScheduler;
class BaseMappingGraphNode;
class MappingGraph;

/**
 * @brief Provides graph traversal algorithms for the mapping graph.
 *
 * This class contains static and instance methods that implement various ways
 * to traverse the nodes and component groups defined in a MappingGraph,
 * such as top-down, bottom-up, or arbitrary order. It uses the Visitor pattern
 * to process components during traversal.
 */
struct SOFA_SIMULATION_CORE_API MappingGraphAlgorithms
{
    /**
     * @brief Constructor for MappingGraphAlgorithms.
     * @param mappingGraph Pointer to the mapping graph to be traversed.
     */
    explicit MappingGraphAlgorithms(MappingGraph* mappingGraph)
        : m_mappingGraph(mappingGraph)
    {
    }

    // ------------------------------------------------------------------
    // Traverse without any specific order
    // ------------------------------------------------------------------

    /**
     * @brief Traverses the entire mapping graph nodes in an arbitrary order.
     * @param visitor The concrete visitor implementation to process each node.
     * @param scope Specifies which types of nodes should be visited (e.g., all, or only component groups).
     */
    void traverse(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    void traverse(MappingGraphVisitor& visitor, VisitorApplication scope, TaskScheduler* taskScheduler) const;

    /**
     * @brief Traverses the entire mapping graph nodes using a callable function in an arbitrary order.
     * @tparam Callable The type of callable object used for visitation.
     * @param callable The callable object containing the logic to execute during traversal.
     * @param scope Specifies which types of nodes should be visited.
     */
    template<class Callable>
    void traverse_(const Callable& callable, VisitorApplication scope = VisitorApplication::ALL_NODES) const
    {
        CallableVisitor<Callable> visitor{callable};
        traverse(visitor, scope);
    }

    // ------------------------------------------------------------------
    // Top-Down traversal
    // ------------------------------------------------------------------

    /**
     * @brief Traverses the mapping graph in a top-down order.
     * @param visitor The concrete visitor implementation to process each node.
     * @param scope Specifies which types of nodes should be visited.
     */
    void traverseTopDown(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    /**
     * @brief Traverses the mapping graph in a top-down order using a callable function.
     * @tparam Callable The type of callable object used for visitation.
     * @param callable The callable object containing the logic to execute during traversal.
     * @param scope Specifies which types of nodes should be visited.
     */
    template<class Callable>
    void traverseTopDown_(const Callable& callable, VisitorApplication scope = VisitorApplication::ALL_NODES) const
    {
        CallableVisitor<Callable> visitor{callable};
        traverseTopDown(visitor, scope);
    }

    // ------------------------------------------------------------------
    // Bottom-Up traversal
    // ------------------------------------------------------------------

    /**
     * @brief Traverses the mapping graph in a bottom-up order.
     * @param visitor The concrete visitor implementation to process each node.
     * @param scope Specifies which types of nodes should be visited.
     */
    void traverseBottomUp(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    /**
     * @brief Traverses the mapping graph in a bottom-up order using a callable function.
     * @tparam Callable The type of callable object used for visitation.
     * @param callable The callable object containing the logic to execute during traversal.
     * @param scope Specifies which types of nodes should be visited.
     */
    template<class Callable>
    void traverseBottomUp_(const Callable& callable, VisitorApplication scope = VisitorApplication::ALL_NODES) const
    {
        CallableVisitor<Callable> visitor{callable};
        traverseBottomUp(visitor, scope);
    }

    // ------------------------------------------------------------------
    // Traverse only component groups without any specific order
    // ------------------------------------------------------------------

    /**
     * @brief Visit and process component groups without any specific order.
     * @param visitor The concrete visitor implementation.
     * @param scope Specifies which types of nodes should be visited. Defaults to ALL_NODES.
     */
    void traverseComponentGroups(MappingGraphVisitor& visitor, VisitorApplication scope = VisitorApplication::ALL_NODES) const;

    /**
     * @brief Visit and process component groups without any specific order using a callable function.
     * @tparam Callable The type of callable object used for visitation.
     * @param callable The callable object containing the logic to execute during traversal.
     * @param scope Specifies which types of nodes should be visited. Defaults to ALL_NODES.
     */
    template<class Callable>
    void traverseComponentGroups_(const Callable& callable, VisitorApplication scope = VisitorApplication::ALL_NODES) const
    {
        CallableVisitor<Callable> visitor{callable};
        traverseComponentGroups(visitor, scope);
    }

    /**
     * @brief Visit and process component groups without any specific order, optionally coordinating
     * with a TaskScheduler to manage execution parallelism.
     * @param visitor The concrete visitor implementation.
     * @param taskScheduler Optional scheduler instance for tasks requiring explicit ordering.
     */
    void traverseComponentGroups(MappingGraphVisitor& visitor, TaskScheduler* taskScheduler) const;

private:
    MappingGraph* m_mappingGraph { nullptr };

    std::queue<BaseMappingGraphNode*> prepareRootForTraversal() const;

    template<class Callable>
    static void processQueue(std::queue<BaseMappingGraphNode*>& ready, const Callable& f);
};
}
