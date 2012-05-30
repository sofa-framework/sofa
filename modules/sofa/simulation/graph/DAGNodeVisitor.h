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
#ifndef SOFA_SIMULATION_GRAPH_DAGNODEVISITOR_H
#define SOFA_SIMULATION_GRAPH_DAGNODEVISITOR_H

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/graph/DAGNode.h>

namespace sofa
{

namespace simulation
{

namespace graph
{

/**
Base class for the Visitors which deal with GNodes specifically rather than Node.

	@author The SOFA team </www.sofa-framework.org>
*/
class SOFA_SIMULATION_GRAPH_API DAGNodeVisitor : public sofa::simulation::Visitor
{
public:
    DAGNodeVisitor(const sofa::core::ExecParams* params);

    ~DAGNodeVisitor();

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(DAGNode* /*node*/) { return RESULT_CONTINUE; }

    /// Callback method called after child node have been processed and before going back to the parent node.
    virtual void processNodeBottomUp(DAGNode* /*node*/) { }

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    /// This version is offered a LocalStorage to store temporary data
    virtual Result processNodeTopDown(DAGNode* node, LocalStorage*) { return processNodeTopDown(node); }

    /// Callback method called after child node have been processed and before going back to the parent node.
    /// This version is offered a LocalStorage to store temporary data
    virtual void processNodeBottomUp(DAGNode* node, LocalStorage*) { processNodeBottomUp(node); }

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(simulation::Node* node)
    {
        DAGNode* g = dynamic_cast<DAGNode*>(node);
        if (!g)
        {
            std::cerr << "DAGNodeVisitor: node is not a DAGNode !\n";
            return RESULT_PRUNE;
        }
        else
        {
            return processNodeTopDown(g);
        }
    }

    /// Callback method called after child node have been processed and before going back to the parent node.
    virtual void processNodeBottomUp(simulation::Node* node)
    {
        DAGNode* g = dynamic_cast<DAGNode*>(node);
        if (!g)
        {
            std::cerr << "DAGNodeVisitor: node is not a DAGNode !\n";
        }
        else
        {
            processNodeBottomUp(g);
        }
    }

    virtual const char* getClassName() const { return "DAGNodeVisitor"; }
    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Visit, class Container, class Object >
    void for_each(Visit* visitor, DAGNode* ctx, const Container& list, void (Visit::*fn)(DAGNode*, Object*))
    {
        for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
        {
            typename Container::pointed_type* ptr = &*(*it);
            if(testTags(ptr))
            {
                debug_write_state_before(ptr);
                ctime_t t=begin(ctx, ptr);
                (visitor->*fn)(ctx, ptr);
                end(ctx, ptr, t);
                debug_write_state_after(ptr);
            }
        }
    }

    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Visit, class Container, class Object >
    Visitor::Result for_each_r(Visit* visitor, DAGNode* ctx, const Container& list, Visitor::Result (Visit::*fn)(DAGNode*, Object*))
    {
        Visitor::Result res = Visitor::RESULT_CONTINUE;
        for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
        {
            typename Container::pointed_type* ptr = &*(*it);
            if(testTags(ptr))
            {
                debug_write_state_before(ptr);
                ctime_t t=begin(ctx, ptr);
                res = (visitor->*fn)(ctx, ptr);
                end(ctx, ptr, t);
                debug_write_state_after(ptr);
            }
        }
        return res;
    }

};

}

}

}

#endif
