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
#include <sofa/simulation/common/Visitor.h>
#include "BglScene.h"
#include <sofa/core/objectmodel/ClassInfo.h>

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
    typedef sofa::simulation::Visitor Visitor;

    /**
    \param sg the SOFA scene containing a bgl graph
    \param n the node of the bgl graph corresponding to this
    */
    BglNode(BglScene* sg, BglScene::Hgraph* g,  BglScene::Hvertex n, const std::string& name="" );
    ~BglNode();

    /** Perform a scene graph traversal with the given Visitor, starting from this node.
    Visitor::processNodetopdown is applied on discover, and Visitor::processNodeBottomUp is applied on finish.
    */
    void doExecuteVisitor( Visitor* action);


    // to move to simulation::Node
    void clearInteractionForceFields();

    /// Generic list of objects access, possibly searching up or down from the current context
    /// @todo Would better be a member of BglScene
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const;



protected:
    BglScene* scene;
    BglScene::Hgraph* graph;      ///< the graph it is inserted to
    BglScene::Hvertex vertexId;  ///< its id in the graph

};

}
}
}

#endif
