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


    // to move to simulation::Node
    void clearInteractionForceFields();

    /// Generic list of objects access, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir = SearchUp) const;



protected:
    BglScene* scene;
    BglScene::Hvertex vertexId;  ///< represents this in graph.bglGraph

};

}
}
}

#endif
