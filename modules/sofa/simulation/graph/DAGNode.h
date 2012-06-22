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
#ifndef SOFA_SIMULATION_GRAPH_DAGNODE_H
#define SOFA_SIMULATION_GRAPH_DAGNODE_H

#include <sofa/simulation/graph/graph.h>
#include <sofa/simulation/common/Node.h>



namespace sofa
{

namespace simulation
{
namespace graph
{


/** Define the structure of the scene. Contains (as pointer lists) Component objects and children DAGNode objects.
 */
class SOFA_SIMULATION_GRAPH_API DAGNode : public simulation::Node
{
public:
    typedef Node::DisplayFlags DisplayFlags;
    SOFA_CLASS(DAGNode, simulation::Node);

protected:
    DAGNode( const std::string& name="", DAGNode* parent=NULL  );

    virtual ~DAGNode();

public:
    //Pure Virtual method from Node
    virtual Node::SPtr createChild(const std::string& nodeName);

    //Pure Virtual method from BaseNode
    /// Add a child node
    virtual void addChild(BaseNode::SPtr node);

    /// Remove a child node
    virtual void removeChild(BaseNode::SPtr node);

    /// Move a node from another node
    virtual void moveChild(BaseNode::SPtr obj);

    /// Add an object and return this. Detect the implemented interfaces and add the object to the corresponding lists.
    virtual bool addObject(core::objectmodel::BaseObject::SPtr obj) { return simulation::Node::addObject(obj); }

    /// Remove an object
    virtual bool removeObject(core::objectmodel::BaseObject::SPtr obj) { return simulation::Node::removeObject(obj); }

    /// Remove the current node from the graph: consists in removing the link to its parent
    virtual void detachFromGraph();

    /// Get a list of parent node
    virtual Parents getParents() const;

    /// Test if the given node is a parent of this node.
    bool hasParent(const BaseNode* node) const;

    /// Test if the given context is a parent of this context.
    bool hasParent(const BaseContext* context) const;

    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseNode* node) const
    {
        return hasAncestor(node->getContext());
    }

    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseContext* context) const;


    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const;

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const;






    /// Called during initialization to corectly propagate the visual context to the children
    virtual void initVisualContext();

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext();

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext();


    /// Return the full path name of this node
    std::string getPathName() const;

    static DAGNode::SPtr create(DAGNode*, xml::Element<core::objectmodel::BaseNode>* arg)
    {
        DAGNode::SPtr obj = DAGNode::SPtr();
        obj->parse(arg);
        return obj;
    }

    DAGNode* parent() const { return l_parent.get(); }

protected:

    SingleLink<DAGNode,DAGNode,BaseLink::FLAG_DOUBLELINK> l_parent;

    virtual void doAddChild(DAGNode::SPtr node);
    void doRemoveChild(DAGNode::SPtr node);


    /// Execute a recursive action starting from this node.
    /// This method bypass the actionScheduler of this node if any.
    void doExecuteVisitor(simulation::Visitor* action);
    // VisitorScheduler can use doExecuteVisitor() method
    friend class simulation::VisitorScheduler;


};

} // namespace graph

} // namespace simulation

} // namespace sofa

#endif
