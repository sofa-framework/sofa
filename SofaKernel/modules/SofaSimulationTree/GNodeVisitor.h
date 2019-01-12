/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_TREE_GNODEVISITOR_H
#define SOFA_SIMULATION_TREE_GNODEVISITOR_H

#include <sofa/simulation/Visitor.h>
#include <SofaSimulationTree/GNode.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

/**
Base class for the Visitors which deal with GNodes specifically rather than Node.

    @author The SOFA team </www.sofa-framework.org>
*/
class SOFA_SIMULATION_TREE_API GNodeVisitor : public Visitor
{
public:
    GNodeVisitor(const sofa::core::ExecParams* params);

    ~GNodeVisitor();

    using Visitor::processNodeTopDown;
    using Visitor::processNodeBottomUp;

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(GNode* /*node*/) { return RESULT_CONTINUE; }

    /// Callback method called after child node have been processed and before going back to the parent node.
    virtual void processNodeBottomUp(GNode* /*node*/) { }

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    /// This version is offered a LocalStorage to store temporary data
    virtual Result processNodeTopDown(GNode* node, LocalStorage*) { return processNodeTopDown(node); }

    /// Callback method called after child node have been processed and before going back to the parent node.
    /// This version is offered a LocalStorage to store temporary data
    virtual void processNodeBottomUp(GNode* node, LocalStorage*) { processNodeBottomUp(node); }

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(simulation::Node* node)
    {
        GNode* g = dynamic_cast<GNode*>(node);
        if (!g)
        {
            dmsg_error("GNodeVisitor") << "Node is not a GNode.";
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
        GNode* g = dynamic_cast<GNode*>(node);
        if (!g)
        {
            dmsg_error("GNodeVisitor") << "Node is not a GNode.";
        }
        else
        {
            processNodeBottomUp(g);
        }
    }

    virtual const char* getClassName() const { return "GNodeVisitor"; }
    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Visit, class Container, class Object >
    void for_each(Visit* visitor, GNode* ctx, const Container& list, void (Visit::*fn)(GNode*, Object*))
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
    Visitor::Result for_each_r(Visit* visitor, GNode* ctx, const Container& list, Visitor::Result (Visit::*fn)(GNode*, Object*))
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
