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
#ifndef sofa_simulation_treeGNodeVisitor_h
#define sofa_simulation_treeGNodeVisitor_h

#include <sofa/simulation/tree/Visitor.h>
#include <sofa/simulation/tree/GNode.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

/**
Base class for the Visitors which deal with GNodes specifically rather than System.

	@author The SOFA team </www.sofa-framework.org>
*/
class GNodeVisitor : public sofa::simulation::tree::Visitor
{
public:
    GNodeVisitor();

    ~GNodeVisitor();

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
