/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_SIMULATION_TREE_GNODE_H
#define SOFA_SIMULATION_TREE_GNODE_H

#include <SofaSimulationTree/tree.h>
#include <sofa/simulation/Node.h>



namespace sofa
{

namespace simulation
{
namespace tree
{


/** Define the structure of the scene. Contains (as pointer lists) Component objects and children GNode objects.
 */
class SOFA_SIMULATION_TREE_API GNode : public simulation::Node
{
public:
    typedef Node::DisplayFlags DisplayFlags;
    SOFA_CLASS(GNode, simulation::Node);

protected:
    GNode( const std::string& name="", GNode* parent=NULL  );

    virtual ~GNode();

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

    /// returns number of parents
    virtual size_t getNbParents() const;

    /// return the first parent (returns NULL if no parent)
    virtual BaseNode* getFirstParent() const;

    /// Test if the given node is a parent of this node.
    bool hasParent(const BaseNode* node) const
    {
        return parent() == node;
    }

    /// Test if the given context is a parent of this context.
    bool hasParent(const BaseContext* context) const
    {
        if (context == NULL) return parent() == NULL;
        else return parent()->getContext() == context;
    }

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

    static GNode::SPtr create(GNode*, core::objectmodel::BaseObjectDescription* arg)
    {
        GNode::SPtr obj = GNode::SPtr();
        obj->parse(arg);
        return obj;
    }


    /// return the smallest common parent between this and node2 (returns NULL if separated sub-graphes)
    virtual Node* findCommonParent( simulation::Node* node2 );

protected:

    inline GNode* parent() const { return l_parent.get(); }

    SingleLink<GNode,GNode,BaseLink::FLAG_DOUBLELINK> l_parent;

    virtual void doAddChild(GNode::SPtr node);
    void doRemoveChild(GNode::SPtr node);


    /// Execute a recursive action starting from this node.
    void doExecuteVisitor(simulation::Visitor* action, bool=false);
    // VisitorScheduler can use doExecuteVisitor() method
    friend class simulation::VisitorScheduler;


};

} // namespace tree

} // namespace simulation

} // namespace sofa

#endif
