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
#ifndef SOFA_SIMULATION_GRAPH_DAGNODE_H
#define SOFA_SIMULATION_GRAPH_DAGNODE_H

#include <SofaSimulationGraph/graph.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/objectmodel/Link.h>
#include <sofa/simulation/Visitor.h>

namespace sofa
{

namespace simulation
{
namespace graph
{

/** Define the structure of the scene as a Directed Acyclic Graph. Contains component objects (as pointer lists) and parents/childs (as DAGNode objects).
 *
 * The visitor traversal is performed in two passes:
 *      - a complete top-down traversal
 *      - then a complete bottom-up traversal in the exact invert order than the top-down traversal
 * NB: contrary to the "tree" traversal, there are no interlinked forward/backward callbacks. There are only forward then only backward callbacks.
 *
 * Note that nodes created during a traversal are not traversed if they are created upper than the current node during the top-down traversal or if they are created during the bottom-up traversal.
 */
class SOFA_SIMULATION_GRAPH_API DAGNode : public simulation::Node
{
public:
    typedef Node::DisplayFlags DisplayFlags;
    SOFA_CLASS(DAGNode, simulation::Node);

    typedef MultiLink<DAGNode,DAGNode,BaseLink::FLAG_STOREPATH|BaseLink::FLAG_DOUBLELINK> LinkParents;
    typedef LinkParents::const_iterator ParentIterator;


protected:
    DAGNode( const std::string& name="", DAGNode* parent=nullptr  );

    virtual ~DAGNode() override;

public:
    /// Pure Virtual method from Node
    virtual Node::SPtr createChild(const std::string& nodeName) override;

    /// Remove the current node from the graph: consists in removing the link to its parent
    virtual void detachFromGraph() override;

    /// Get a list of parent node
    virtual Parents getParents() const override;

    /// returns number of parents
    virtual size_t getNbParents() const override;

    /// return the first parent (returns NULL if no parent)
    virtual BaseNode* getFirstParent() const override;

    /// Test if the given node is a parent of this node.
    bool hasParent(const BaseNode* node) const override;

    /// Test if the given context is a parent of this context.
    bool hasParent(const BaseContext* context) const;

    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseNode* node) const override
    {
        return hasAncestor(node->getContext());
    }

    /// Test if the given context is an ancestor of this context.
    /// An ancestor is a parent or (recursively) the parent of an ancestor.
    bool hasAncestor(const BaseContext* context) const override;


    /// Generic object access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const override;

    /// Generic object access, given a path from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void* getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const override;

    /// Generic list of objects access, given a set of required tags, possibly searching up or down from the current context
    ///
    /// Note that the template wrapper method should generally be used to have the correct return type,
    virtual void getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir = SearchUp) const override;


    /// Mesh Topology that is relevant for this context
    /// (within it or its parents until a mapping is reached that does not preserve topologies).
    virtual core::topology::BaseMeshTopology* getActiveMeshTopology() const override;


    /// Called during initialization to corectly propagate the visual context to the children
    virtual void initVisualContext() override;

    /// Update the whole context values, based on parent and local ContextObjects
    virtual void updateContext() override;

    /// Update the simulation context values(gravity, time...), based on parent and local ContextObjects
    virtual void updateSimulationContext() override;

    static DAGNode::SPtr create(DAGNode*, core::objectmodel::BaseObjectDescription* arg)
    {
        DAGNode::SPtr obj = DAGNode::SPtr();
        obj->parse(arg);
        return obj;
    }


    /// return the smallest common parent between this and node2 (returns NULL if separated sub-graphes)
    virtual Node* findCommonParent( Node* node2 ) override;

    /// compute the traversal order from this Node
    virtual void precomputeTraversalOrder( const core::ExecParams* params ) override;

protected:

    /// bottom-up traversal, returning the first node which have a descendancy containing both node1 & node2
    DAGNode* findCommonParent( DAGNode* node1, DAGNode* node2 );


    LinkParents l_parents;

    virtual void doAddChild(BaseNode::SPtr node) override;
    virtual void doRemoveChild(BaseNode::SPtr node) override;
    virtual void doMoveChild(BaseNode::SPtr node, BaseNode::SPtr previous_parent) override;


    /// Execute a recursive action starting from this node.
    void doExecuteVisitor(simulation::Visitor* action, bool precomputedOrder=false) override;


    /// @name @internal stuff related to the DAG traversal
    /// @{


    /// all child nodes (unordered)
    std::set<DAGNode*> _descendancy;

    /// bottom-up traversal removing descendancy
    void setDirtyDescendancy();

    /// traversal updating the descendancy
    void updateDescendancy();

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

    /// the ordered list of Node to traverse from this Node
    NodeList _precomputedTraversalOrder;

    /// @internal performing only the top-down traversal on a DAG
    /// @executedNodes will be fill with the DAGNodes where the top-down action is processed
    /// @statusMap the visitor's flag map
    /// @visitorRoot node from where the visitor has been run
    void executeVisitorTopDown(simulation::Visitor* action, NodeList& executedNodes, StatusMap& statusMap, DAGNode* visitorRoot );
    void executeVisitorBottomUp(simulation::Visitor* action, NodeList& executedNodes );
    /// @}

    /// @internal tree traversal implementation
    void executeVisitorTreeTraversal( Visitor* action, StatusMap& statusMap, Visitor::TreeTraversalRepetition repeat, bool alreadyRepeated=false );

    /// @name @internal stuff related to getObjects
    /// @{

    /// get node's local objects respecting specified class_info and tags
    void getLocalObjects( const sofa::core::objectmodel::ClassInfo& class_info, DAGNode::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags ) const ;

    friend class GetDownObjectsVisitor ;
    friend class GetUpObjectsVisitor ;
    /// @}
};

/// get all down objects respecting specified class_info and tags
class GetDownObjectsVisitor : public Visitor
{
public:

    GetDownObjectsVisitor(const sofa::core::objectmodel::ClassInfo& class_info, DAGNode::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags);

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const;
    virtual const char* getClassName()    const;


protected:

    const sofa::core::objectmodel::ClassInfo& _class_info;
    DAGNode::GetObjectsCallBack& _container;
    const sofa::core::objectmodel::TagSet& _tags;
};

/// get all up objects respecting specified class_info and tags
class GetUpObjectsVisitor : public Visitor
{
public:
    GetUpObjectsVisitor(DAGNode* searchNode, const sofa::core::objectmodel::ClassInfo& class_info, DAGNode::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags);

    virtual Result processNodeTopDown(simulation::Node* node);

    /// Specify whether this action can be parallelized.
    virtual bool isThreadSafe() const;

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    virtual const char* getCategoryName() const;
    virtual const char* getClassName()    const;


protected:
    DAGNode* _searchNode;
    const sofa::core::objectmodel::ClassInfo& _class_info;
    DAGNode::GetObjectsCallBack& _container;
    const sofa::core::objectmodel::TagSet& _tags;

};

} // namespace graph

} // namespace simulation

} // namespace sofa

#endif

