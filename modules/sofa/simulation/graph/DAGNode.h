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
#include <sofa/core/objectmodel/Link.h>
using namespace sofa::core::objectmodel;


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

    typedef MultiLink<DAGNode,DAGNode,BaseLink::FLAG_STOREPATH|BaseLink::FLAG_DOUBLELINK> LinkParents;
    typedef LinkParents::const_iterator ParentIterator;


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

protected:

    LinkParents l_parents;

    virtual void doAddChild(DAGNode::SPtr node);
    void doRemoveChild(DAGNode::SPtr node);


    /// Execute a recursive action starting from this node.
    /// This method bypass the actionScheduler of this node if any.
    void doExecuteVisitor(simulation::Visitor* action);



    /// @name @internal stuff related to the DAG traversal
    /// @{


    /// all child nodes (unordered)
    std::set<DAGNode*> _descendancy;

    /// bottom-up traversal removing descendancy
    void setDirtyDescendancy();

    /// traversal updating the descendancy
    void updateDescendancy();


    // need to update the ancestor descendancy
    virtual void notifyAddChild(Node::SPtr node);
    // need to update the ancestor descendancy
    virtual void notifyRemoveChild(Node::SPtr node);
    // need to update the ancestor descendancy
    virtual void notifyMoveChild(Node::SPtr node, Node* prev);



    /// traversal flags
    typedef enum
    {
        NOT_VISITED=0,
        VISITED,
        PRUNED
    } VisitedStatus;


    /// wrapper to use VisitedStatus in a std::map (to ensure the default map insertion will give NOT_VISITED)
    struct StatusStruct
    {
        StatusStruct() : status(NOT_VISITED) {}
        StatusStruct( const VisitedStatus& s ) : status(s) {}
        inline void operator=( const VisitedStatus& s ) { status=s; }
        inline bool operator==( const VisitedStatus& s ) { return status==s; }
        inline bool operator==( const StatusStruct& s ) { return status==s.status; }
        inline bool operator!=( const VisitedStatus& s ) { return status!=s; }
        inline bool operator!=( const StatusStruct& s ) { return status!=s.status; }
        VisitedStatus status;
    };

    /// map structure to store a traversal flag for each DAGNode
    typedef std::map<DAGNode*,StatusStruct> StatusMap;

    /// list of DAGNode*
    typedef std::list<DAGNode*> NodeList;

    /// @internal performing only the top-down traversal on a DAG
    /// @executedNodes will be fill with the DAGNodes where the top-down action is processed
    /// @statusMap the visitor's flag map
    /// @visitorRoot node from where the visitor has been run
    void executeVisitorTopDown(simulation::Visitor* action, NodeList& executedNodes, StatusMap& statusMap, DAGNode* visitorRoot );

    /// @}


};

} // namespace graph

} // namespace simulation

} // namespace sofa

#endif
