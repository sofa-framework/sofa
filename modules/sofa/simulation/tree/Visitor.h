/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#ifndef SOFA_SIMULATION_TREE_ACTION_H
#define SOFA_SIMULATION_TREE_ACTION_H

#include <sofa/component/System.h>
#include <sofa/simulation/tree/LocalStorage.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

class component::System;
class LocalStorage;

/// Base class for actions propagated recursively through the scenegraph
class Visitor
{
public:
    virtual ~Visitor() {}

    enum Result { RESULT_CONTINUE, RESULT_PRUNE };

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    virtual Result processNodeTopDown(component::System* /*node*/) { return RESULT_CONTINUE; }

    /// Callback method called after child node have been processed and before going back to the parent node.
    virtual void processNodeBottomUp(component::System* /*node*/) {}

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const { return "default"; }

    /// Helper method to enumerate objects in the given list. The callback gets the pointer to node
    template < class Act, class Container, class Object >
    void for_each(Act* action, component::System* node, const Container& list, void (Act::*fn)(component::System*, Object*))
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
    Visitor::Result for_each_r(Act* action, component::System* node, const Container& list, Visitor::Result (Act::*fn)(component::System*, Object*))
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

    //template < class Act, class Container, class Object >
    //void for_each(Act* action, const Container& list, void (Act::*fn)(Object))
    //{
    //	for (typename Container::iterator it=list.begin(); it != list.end(); ++it)
    //	{
    //		(action->*fn)(*it);
    //	}
    //}

    typedef component::System::ctime_t ctime_t;

    /// Optional helper method to call before handling an object if not using the for_each method.
    /// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
    ctime_t begin(component::System* node, core::objectmodel::BaseObject* /*obj*/)
    {
        return node->startTime();
    }

    /// Optional helper method to call after handling an object if not using the for_each method.
    /// It currently takes care of time logging, but could be extended (step-by-step execution for instance)
    void end(component::System* node, core::objectmodel::BaseObject* obj, ctime_t t0)
    {
        node->endTime(t0, getCategoryName(), obj);
    }

    /// Alias for context->executeVisitor(this)
    void execute(core::objectmodel::BaseContext*);


    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const { return false; }

    /// Callback method called when decending to a new node. Recursion will stop if this method returns RESULT_PRUNE
    /// This version is offered a LocalStorage to store temporary data
    virtual Result processNodeTopDown(component::System* node, LocalStorage*) { return processNodeTopDown(node); }

    /// Callback method called after child node have been processed and before going back to the parent node.
    /// This version is offered a LocalStorage to store temporary data
    virtual void processNodeBottomUp(component::System* node, LocalStorage*) { processNodeBottomUp(node); }
};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
