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
#include <sofa/simulation/MappingGraph.h>
#include <sofa/simulation/mappinggraph/MappingGraphAlgorithms.h>
#include <sofa/simulation/task/ParallelForEach.h>

namespace sofa::simulation
{

namespace
{
bool shouldVisit(const BaseMappingGraphNode* node, VisitorApplication scope)
{
    if (node)
    {
        switch (scope)
        {
            case VisitorApplication::ONLY_MAPPED_NODES:
                return node->isMapped();
            case VisitorApplication::ONLY_MAIN_NODES:
                return !node->isMapped();
            case VisitorApplication::ALL_NODES:
            default:
                return true; // Visit all nodes.
        }
    }
    return false;
}
}

std::queue<BaseMappingGraphNode*> MappingGraphAlgorithms::prepareRootForTraversal() const
{
    std::queue<BaseMappingGraphNode*> ready;
    for (auto& node : m_mappingGraph->m_allNodes)
    {
        node->m_pendingCount = static_cast<int>(node->m_parents.size());
        if (node->m_pendingCount == 0) //node without any parent -> a root
        {
            ready.push(node.get());
        }
    }

    return ready;
}


/**
 * @brief Performs a breadth-first search (BFS) traversal, processing nodes in dependency order.
 *
 * This static helper method is used for both top-down and bottom-up traversals.
 *
 * @param ready The queue of nodes that are currently ready to be visited/processed.
 */
template<class Callable>
void MappingGraphAlgorithms::processQueue(std::queue<BaseMappingGraphNode*>& ready, const Callable& f)
{
    while (!ready.empty())
    {
        BaseMappingGraphNode* current = ready.front();
        ready.pop();

        f(current);

        for (auto& child : current->m_children)
        {
            --(child->m_pendingCount);
            if (child->m_pendingCount == 0)
            {
                ready.push(child.get());
            }
        }
    }
}


void MappingGraphAlgorithms::traverse(MappingGraphVisitor& visitor, VisitorApplication scope) const
{
    for (auto& node : m_mappingGraph->m_allNodes)
    {
        if (shouldVisit(node.get(), scope))
        {
            node->accept(visitor);
        }
    }
}

void MappingGraphAlgorithms::traverse(MappingGraphVisitor& visitor,
    VisitorApplication scope, TaskScheduler* taskScheduler) const
{
    if (taskScheduler)
    {
        sofa::type::vector<BaseMappingGraphNode::SPtr> visitableNodes;
        for (const auto& node : m_mappingGraph->m_allNodes)
        {
            if (shouldVisit(node.get(), scope))
            {
                visitableNodes.push_back(node);
            }
        }

        sofa::simulation::parallelForEach(*taskScheduler,
            visitableNodes.begin(), visitableNodes.end(),
            [&visitor](const auto& node)
            {
                node->accept(visitor);
            });
    }
    else
    {
        traverse(visitor);
    }
}

void MappingGraphAlgorithms::traverseTopDown(MappingGraphVisitor& visitor,
                                             VisitorApplication scope) const
{
    std::queue<BaseMappingGraphNode*> ready = prepareRootForTraversal();
    processQueue(ready, [&visitor, scope](const BaseMappingGraphNode* node)
    {
        if (shouldVisit(node, scope))
        {
            node->accept(visitor);
        }
    });
}

void MappingGraphAlgorithms::traverseBottomUp(MappingGraphVisitor& visitor,
                                              VisitorApplication scope) const
{
    //the strategy consists in traversing the graph from top to bottom and
    //register the traversed nodes in a list. The bottom-up traversal corresponds to the
    //reversed list.

    sofa::type::vector<BaseMappingGraphNode*> nodes;
    nodes.reserve(m_mappingGraph->m_allNodes.size());
    {
        std::queue<BaseMappingGraphNode*> ready = prepareRootForTraversal();
        processQueue(ready, [&nodes, scope](BaseMappingGraphNode* node)
        {
            if (shouldVisit(node, scope))
            {
                nodes.push_back(node);
            }
        });
    }

    for (auto it = nodes.crbegin(); it != nodes.crend(); ++it)
    {
        (*it)->accept(visitor);
    }
}

void MappingGraphAlgorithms::traverseComponentGroups(
    MappingGraphVisitor& visitor, VisitorApplication scope) const
{
    for (auto& [states, node] : m_mappingGraph->m_groupIndex)
    {
        if (shouldVisit(node.get(), scope))
        {
            for (auto& child : node->m_children)
            {
                child->accept(visitor);
            }
        }
    }
}

void MappingGraphAlgorithms::traverseComponentGroups(MappingGraphVisitor& visitor,
                                                     VisitorApplication scope,
                                                     TaskScheduler* taskScheduler) const
{
    if (taskScheduler)
    {
        sofa::type::vector<BaseMappingGraphNode::SPtr> parallelNodes, sequentialNodes;
        for (const auto& [states, node] : m_mappingGraph->m_groupIndex)
        {
            if (shouldVisit(node.get(), scope))
            {
                //with a size of 1, we are sure that they are all different, preventing data races
                if ( states.size() == 1)
                {
                    parallelNodes.push_back(node);
                }
                else
                {
                    sequentialNodes.push_back(node);
                }
            }
        }

        sofa::simulation::parallelForEach(*taskScheduler,
            parallelNodes.begin(), parallelNodes.end(),
            [&visitor, &scope](const auto& node)
            {
                for (auto& child : node->m_children)
                {
                    child->accept(visitor);
                }
            });

        for (const auto& node : sequentialNodes)
        {
            for (auto& child : node->m_children)
            {
                child->accept(visitor);
            }
        }
    }
    else
    {
        traverseComponentGroups(visitor);
    }
}


}  // namespace sofa::simulation
