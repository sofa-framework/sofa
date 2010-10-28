/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
//
// C++ Interface: GNodeVisitor
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef sofa_simulation_tree_GNodeVisitor_h
#define sofa_simulation_tree_GNodeVisitor_h

#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/tree/GNode.h>

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
class SOFA_SIMULATION_TREE_API GNodeVisitor : public sofa::simulation::Visitor
{
public:
    GNodeVisitor(const sofa::core::ExecParams* params);

    ~GNodeVisitor();

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
            std::cerr << "GNodeVisitor: node is not a GNode !\n";
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
            std::cerr << "GNodeVisitor: node is not a GNode !\n";
        }
        else
        {
            processNodeBottomUp(g);
        }
    }

    virtual const char* getClassName() const { return "GNodeVisitor"; }
    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Act, class Container, class Object >
    void for_each(Act* action, GNode* node, const Container& list, void (Act::*fn)(GNode*, Object*))
    {
        if (node->getLogTime())
        {
            const std::string category = getCategoryName();
            ctime_t t0 = node->startTime();
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                (action->*fn)(node, *it);
                t0 = node->endTime(t0, category, *it);
            }
        }
        else
        {
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                (action->*fn)(node, *it);
            }
        }
    }

    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Act, class Container, class Object >
    Visitor::Result for_each_r(Act* action, GNode* node, const Container& list, Visitor::Result (Act::*fn)(GNode*, Object*))
    {
        Visitor::Result res = Visitor::RESULT_CONTINUE;
        if (node->getLogTime())
        {
            const std::string category = getCategoryName();
            ctime_t t0 = node->startTime();
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                res = (action->*fn)(node, *it);
                t0 = node->endTime(t0, category, *it);
            }
        }
        else
        {
            for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
            {
                res = (action->*fn)(node, *it);
            }
        }
        return res;
    }

};

}

}

}

#endif
