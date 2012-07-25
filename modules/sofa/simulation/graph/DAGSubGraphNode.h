/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

#ifndef DAGSUBGRAPHNODE_H
#define DAGSUBGRAPHNODE_H

#include <sofa/simulation/common/Node.h>

namespace sofa
{
namespace simulation
{
namespace graph
{

class DAGNode;

/**
   Implements a subset of the graph; each DAGSubGraphNode points to a DAGNode
   @author The SOFA team </www.sofa-framework.org>
 */

class DAGSubGraphNode
{
public:
    DAGSubGraphNode(DAGNode *node);
    ~DAGSubGraphNode();

    enum Direction
    {
        downward,
        upward
    };

    /// a subgraph has only ONE root
    DAGSubGraphNode *getRoot();

    /// is this node in the sub-graph?
    DAGSubGraphNode* findNode(DAGNode *node,Direction direction);

    /// adds a child
    void addChild(DAGSubGraphNode* node);

    /// visitor execution
    void executeVisitor(simulation::Visitor* action);

private:
    DAGNode *_node;

    enum
    {
        NOT_VISITED,
        VISITED,
        PRUNED
    } visitedStatus;

    std::list<DAGSubGraphNode*> parents;
    std::list<DAGSubGraphNode*> children;
    typedef std::list<DAGSubGraphNode*>::iterator NodesIterator;


};

} // namespace graph

} // namespace simulation

} // namespace sofa


#endif // DAGSUBGRAPHNODE_H
