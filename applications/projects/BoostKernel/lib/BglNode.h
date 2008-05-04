//
// C++ Interface: BglNode
//
// Description:
//
//
// Author: The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#ifndef BglNode_h
#define BglNode_h

#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/tree/Visitor.h>
#include "BglScene.h"

namespace sofa
{
namespace simulation
{
namespace bgl
{

/**
sofa::simulation::Node as a node of a BGL scene graph.


	@author Francois Faure in The SOFA team </www.sofa-framework.org>
*/
class BglNode : public sofa::simulation::Node
{
public:
    typedef sofa::simulation::tree::Visitor Visitor;

    /**
    \param sg the SOFA scene containing a bgl graph
    \param n the node of the bgl graph corresponding to this
    */
    BglNode(BglScene* sg, BglScene::Hvertex n, const std::string& name="" );
    ~BglNode();

    /** Perform a scene graph traversal with the given Visitor, starting from this node.
    Visitor::processNodetopdown is applied on discover, and Visitor::processNodeBottomUp is applied on finish.
    */
    void doExecuteVisitor( Visitor* action);

    // debug
    void printComponents();

    // to move to simulation::Node
    void clearInteractionForceFields();

protected:
    BglScene* scene;
    BglScene::Hvertex vertexId;  ///< represents this in graph.bglGraph

};

}
}
}

#endif
