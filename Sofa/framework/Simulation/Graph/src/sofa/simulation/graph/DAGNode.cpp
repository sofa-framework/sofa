/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/graph/DAGNode.h>
#include <sofa/simulation/common/xml/NodeElement.h>
#include <sofa/helper/Factory.inl>
#include <sofa/core/Mapping.h>

namespace sofa::simulation::graph
{

/// get all down objects respecting specified class_info and tags
class GetDownObjectsVisitor : public Visitor
{
public:

    GetDownObjectsVisitor(const sofa::core::objectmodel::ClassInfo& class_info, DAGNode::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags);
    ~GetDownObjectsVisitor() override;

    Result processNodeTopDown(simulation::Node* node) override
    {
        static_cast<const DAGNode*>(node)->getLocalObjects( _class_info, _container, _tags );
        return RESULT_CONTINUE;
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return false; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "GetDownObjectsVisitor"; }
    const char* getClassName()    const override { return "GetDownObjectsVisitor"; }

protected:
    const sofa::core::objectmodel::ClassInfo& _class_info;
    DAGNode::GetObjectsCallBack& _container;
    const sofa::core::objectmodel::TagSet& _tags;
};

GetDownObjectsVisitor::GetDownObjectsVisitor(const sofa::core::objectmodel::ClassInfo& class_info,
                                             DAGNode::GetObjectsCallBack& container,
                                             const sofa::core::objectmodel::TagSet& tags)
    : Visitor( sofa::core::execparams::defaultInstance() )
    , _class_info(class_info)
    , _container(container)
    , _tags(tags)
{}

GetDownObjectsVisitor::~GetDownObjectsVisitor(){}

/// get all up objects respecting specified class_info and tags
class GetUpObjectsVisitor : public Visitor
{
public:

    GetUpObjectsVisitor(DAGNode* searchNode, const sofa::core::objectmodel::ClassInfo& class_info, DAGNode::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags);
    ~GetUpObjectsVisitor() override;

    Result processNodeTopDown(simulation::Node* node) override
    {
        const DAGNode* dagnode = dynamic_cast<const DAGNode*>(node);
        if( dagnode->_descendancy.contains(_searchNode) ) // searchNode is in the current node descendancy, so the current node is a parent of searchNode
        {
            dagnode->getLocalObjects( _class_info, _container, _tags );
            return RESULT_CONTINUE;
        }
        else // the current node is NOT a parent of searchNode, stop here
        {
            return RESULT_PRUNE;
        }
    }

    /// Specify whether this action can be parallelized.
    bool isThreadSafe() const override { return false; }

    /// Return a category name for this action.
    /// Only used for debugging / profiling purposes
    const char* getCategoryName() const override { return "GetUpObjectsVisitor"; }
    const char* getClassName()    const override { return "GetUpObjectsVisitor"; }


protected:

    DAGNode* _searchNode;
    const sofa::core::objectmodel::ClassInfo& _class_info;
    DAGNode::GetObjectsCallBack& _container;
    const sofa::core::objectmodel::TagSet& _tags;

};

GetUpObjectsVisitor::GetUpObjectsVisitor(DAGNode* searchNode,
                                         const sofa::core::objectmodel::ClassInfo& class_info,
                                         DAGNode::GetObjectsCallBack& container,
                                         const sofa::core::objectmodel::TagSet& tags)
    : Visitor( sofa::core::execparams::defaultInstance() )
    , _searchNode( searchNode )
    , _class_info(class_info)
    , _container(container)
    , _tags(tags)
{}

GetUpObjectsVisitor::~GetUpObjectsVisitor(){}

DAGNode::DAGNode(const std::string& name, DAGNode* parent)
    : simulation::Node(name)
    , l_parents(initLink("parents", "Parents nodes in the graph"))
{
    if( parent )
        parent->addChild(dynamic_cast<Node*>(this));
}

DAGNode::~DAGNode()
{
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
    {
        const DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<DAGNode>(*it);
        dagnode->l_parents.remove(this);
    }
}

/// Create, add, then return the new child of this Node
Node::SPtr DAGNode::createChild(const std::string& nodeName)
{
    DAGNode::SPtr newchild;
    if (nodeName.empty())
    {
        int i = 0;
        std::string newName = "unnamed";
        bool uid_found = false;
        while (!uid_found)
        {
            uid_found = true;
            for (const auto& c : this->child)
            {
                if (c->getName() == newName)
                {
                    newName = "unnamed" + std::to_string(++i);
                    uid_found = true;
                }
            }
            for (const auto& o : this->object)
            {
                if (o->getName() == newName)
                {
                    newName = "unnamed" + std::to_string(++i);
                    uid_found = true;
                }
            }
        }
        msg_error("Node::createChild()") << "Empty string given to property 'name': Forcefully setting an empty name is forbidden.\n"
                                        "Renaming to " + newName + " to avoid unexpected behaviors.";
        newchild = sofa::core::objectmodel::New<DAGNode>(newName);
    }
    else
        newchild = sofa::core::objectmodel::New<DAGNode>(nodeName);
    this->addChild(newchild); newchild->updateSimulationContext();
    return newchild;
}


void DAGNode::moveChild(BaseNode::SPtr node)
{
    const DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<DAGNode>(node);
    for (const auto& parent : dagnode->getParents()) {
        Node::moveChild(node, parent);
    }
}


/// Add a child node
void DAGNode::doAddChild(BaseNode::SPtr node)
{
    const DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<DAGNode>(node);
    setDirtyDescendancy();
    child.add(dagnode);
    dagnode->l_parents.add(this);
    dagnode->l_parents.updateLinks(); // to fix load-time unresolved links
}

/// Remove a child
void DAGNode::doRemoveChild(BaseNode::SPtr node)
{
    const DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<DAGNode>(node);
    setDirtyDescendancy();
    child.remove(dagnode);
    dagnode->l_parents.remove(this);
}

/// Move a node from another node
void DAGNode::doMoveChild(BaseNode::SPtr node, BaseNode::SPtr previous_parent)
{
    const DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_static_cast<DAGNode>(node);
    if (!dagnode) return;

    setDirtyDescendancy();
    previous_parent->removeChild(node);

    addChild(node);
}

/// Remove a child
void DAGNode::detachFromGraph()
{
    DAGNode::SPtr me = this; // make sure we don't delete ourself before the end of this method
    const LinkParents::Container& parents = l_parents.getValue();
    while(!parents.empty())
        parents.back()->removeChild(this);
}

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* DAGNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if (dir == SearchRoot)
    {
        if (getNbParents()) return getRootContext()->getObject(class_info, tags, dir);
        else dir = SearchDown; // we are the root, search down from here.
    }
    void *result = nullptr;

    if (dir != SearchParents)
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            sofa::core::objectmodel::BaseObject* obj = it->get();
            if (tags.empty() || (obj)->getTags().includes(tags))
            {

                result = class_info.dynamicCast(obj);
                if (result != nullptr)
                {

                    break;
                }
            }
        }

    if (result == nullptr)
    {
        switch(dir)
        {
            case Local:
                break;
            case SearchParents:
            case SearchUp:
            {
                const LinkParents::Container& parents = l_parents.getValue();
                for ( unsigned int i = 0; i < parents.size() ; ++i){
                    result = parents[i]->getObject(class_info, tags, SearchUp);
                    if (result != nullptr) break;
                }
            }
                break;
            case SearchDown:
                for(ChildIterator it = child.begin(); it != child.end(); ++it)
                {
                    result = (*it)->getObject(class_info, tags, dir);
                    if (result != nullptr) break;
                }
                break;
            case SearchRoot:
                dmsg_error("DAGNode") << "SearchRoot SHOULD NOT BE POSSIBLE HERE.";
                break;
        }
    }

    return result;
}

/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* DAGNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
{
    if (path.empty())
    {
        // local object
        return Node::getObject(class_info, Local);
    }
    else if (path[0] == '/')
    {
        // absolute path; let's start from root
        if (!getNbParents()) return getObject(class_info,std::string(path,1));
        else return getRootContext()->getObject(class_info,path);
    }
    else if (std::string(path,0,2)==std::string("./"))
    {
        std::string newpath = std::string(path, 2);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        return getObject(class_info,newpath);
    }
    else if (std::string(path,0,3)==std::string("../"))
    {
        // tricky case:
        // let's test EACH parent and return the first object found (if any)
        std::string newpath = std::string(path, 3);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        if (getNbParents())
        {
            const LinkParents::Container& parents = l_parents.getValue();
            for ( unsigned int i = 0; i < parents.size() ; ++i)
            {
                void* obj = parents[i]->getObject(class_info,newpath);
                if (obj) return obj;
            }
            return nullptr;   // not found in any parent node at all
        }
        else return getObject(class_info,newpath);
    }
    else
    {
        std::string::size_type pend = path.find('/');
        if (pend == std::string::npos) pend = path.length();
        const std::string name ( path, 0, pend );
        const Node* child = getChild(name);
        if (child)
        {
            while (pend < path.length() && path[pend] == '/')
                ++pend;
            return child->getObject(class_info, std::string(path, pend));
        }
        else if (pend < path.length())
        {
            return nullptr;
        }
        else
        {
            sofa::core::objectmodel::BaseObject* obj = simulation::Node::getObject(name);
            if (obj == nullptr)
            {
                return nullptr;
            }
            else
            {
                void* result = class_info.dynamicCast(obj);
                if (result == nullptr)
                {
                    dmsg_error("DAGNode") << "Object "<<name<<" in "<<getPathName()<<" does not implement class "<<class_info.name() ;
                    return nullptr;
                }
                else
                {
                    return result;
                }
            }
        }
    }
}


/// Generic list of objects access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void DAGNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if( dir == SearchRoot )
    {
        if( getNbParents() )
        {
            getRootContext()->getObjects( class_info, container, tags, dir );
            return;
        }
        else dir = SearchDown; // we are the root, search down from here.
    }


    switch( dir )
    {
        case Local:
            this->getLocalObjects( class_info, container, tags );
            break;

        case SearchUp:
            this->getLocalObjects( class_info, container, tags ); // add locals then SearchParents
            // no break here, we want to execute the SearchParents code.
            [[fallthrough]];
        case SearchParents:
        {
            // a visitor executed from top but only run for this' parents will enforce the selected object unicity due even with diamond graph setups
            GetUpObjectsVisitor vis( const_cast<DAGNode*>(this), class_info, container, tags);
            getRootContext()->executeVisitor(&vis);
        }
            break;

        case SearchDown:
        {
            // a regular visitor is enforcing the selected object unicity
            GetDownObjectsVisitor vis(class_info, container, tags);
            (const_cast<DAGNode*>(this))->executeVisitor(&vis);
            break;
        }
        default:
            break;
    }
}

/// Get a list of parent node
sofa::core::objectmodel::BaseNode::Parents DAGNode::getParents() const
{
    Parents p;

    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
        p.push_back(parents[i]);

    return p;
}


/// returns number of parents
size_t DAGNode::getNbParents() const
{
    return l_parents.getValue().size();
}

/// return the first parent (returns nullptr if no parent)
sofa::core::objectmodel::BaseNode* DAGNode::getFirstParent() const
{
    const LinkParents::Container& parents = l_parents.getValue();
    if( parents.empty() ) return nullptr;
    else return l_parents.getValue()[0];
}


/// Test if the given node is a parent of this node.
bool DAGNode::hasParent(const BaseNode* node) const
{
    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
    {
        if (parents[i]==node) return true;
    }
    return false;
}

/// Test if the given context is a parent of this context.
bool DAGNode::hasParent(const BaseContext* context) const
{
    if (context == nullptr) return !getNbParents();

    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
        if (context == parents[i]->getContext()) return true;
    return false;

}



/// Test if the given context is an ancestor of this context.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool DAGNode::hasAncestor(const BaseContext* context) const
{
    const LinkParents::Container& parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; ++i)
        if (context == parents[i]->getContext()
                || parents[i]->hasAncestor(context))
            return true;
    return false;
}


/// Mesh Topology that is relevant for this context
/// (within it or its parents until a mapping is reached that does not preserve topologies).
sofa::core::topology::BaseMeshTopology* DAGNode::getMeshTopologyLink(SearchDirection dir) const
{
    if (this->meshTopology)
        return this->meshTopology;

    if (dir != Local)
        return Node::getMeshTopologyLink(dir);

    //local case similar to getActiveMeshTopology ...

    // Check if a local mapping stops the search
    if (this->mechanicalMapping && !this->mechanicalMapping->sameTopology())
    {
        return nullptr;
    }
    for ( Sequence<sofa::core::BaseMapping>::iterator i=this->mapping.begin(), iend=this->mapping.end(); i!=iend; ++i )
    {
        if (!(*i)->sameTopology())
        {
            return nullptr;
        }
    }
    // No mapping with a different topology, continue on to the parents
    const LinkParents::Container &parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++ )
    {
        // if the visitor is run from a sub-graph containing a multinode linked with a node outside of the subgraph, do not consider the outside node by looking on the sub-graph descendancy
        if ( parents[i] )
        {
            sofa::core::topology::BaseMeshTopology* res = parents[i]->getMeshTopologyLink(Local);
            if (res)
                return res;
        }
    }
    return nullptr; // not found in any parents
}


void DAGNode::precomputeTraversalOrder( const sofa::core::ExecParams* params )
{
    // accumulating traversed Nodes
    class TraversalOrderVisitor : public Visitor
    {
        NodeList& _orderList;
    public:
        TraversalOrderVisitor(const sofa::core::ExecParams* params, NodeList& orderList )
            : Visitor(params)
            , _orderList( orderList )
        {
            _orderList.clear();
        }

        Result processNodeTopDown(Node* node) override
        {
            _orderList.push_back( static_cast<DAGNode*>(node) );
            return RESULT_CONTINUE;
        }

        const char* getClassName() const override {return "TraversalOrderVisitor";}
    };

    TraversalOrderVisitor tov( params, _precomputedTraversalOrder );
    executeVisitor( &tov, false );
}



/// Execute a recursive action starting from this node
void DAGNode::doExecuteVisitor(simulation::Visitor* action, bool precomputedOrder)
{
    if( precomputedOrder && !_precomputedTraversalOrder.empty() )
    {
        for( NodeList::iterator it = _precomputedTraversalOrder.begin(), itend = _precomputedTraversalOrder.end() ; it != itend ; ++it )
        {
            if ( action->canAccessSleepingNode || !(*it)->getContext()->isSleeping() )
                action->processNodeTopDown( *it );
        }

        for( NodeList::reverse_iterator it = _precomputedTraversalOrder.rbegin(), itend = _precomputedTraversalOrder.rend() ; it != itend ; ++it )
        {
            if ( action->canAccessSleepingNode || !(*it)->getContext()->isSleeping() )
                action->processNodeBottomUp( *it );
        }
    }
    else
    {
        // WARNING: do not store the traversal infos in the DAGNode, as several visitors could traversed the graph simultaneously
        // These infos are stored in a StatusMap per visitor.
        updateDescendancy();

        Visitor::TreeTraversalRepetition repeat;
        if( action->treeTraversal(repeat) )
        {
            // Tree traversal order
            //
            // Diamond shapes are ignored, a child node is visited as soon as a parent node has been visited.
            // The multi-nodes (with several parents) are visited either: only once, only twice or for every times
            // depending on the visitor's 'repeat'
            //
            // Some particular visitors such as a flat graph display or VisualVisitors must follow such a traversal order.

            StatusMap statusMap;
            executeVisitorTreeTraversal( action, statusMap, repeat );
        }
        else
        {
            // Direct acyclic graph traversal order
            //
            // This is the default order, used for mechanics.
            //
            // A child node is visited only when all its parents have been visited.
            // A child node is 'pruned' only if all its parents are 'pruned'.
            // Every executed node in the forward traversal are stored in 'executedNodes',
            // its reverse order is used for the backward traversal.

            // Note that a newly 'pruned' node is still traversed (w/o execution) to be sure to execute its child nodes,
            // that can have ancestors in another branch that is not pruned...
            // An already pruned node is ignored.

            NodeList executedNodes;
            {
                StatusMap statusMap;
                executeVisitorTopDown( action, executedNodes, statusMap, this );
            }
            executeVisitorBottomUp( action, executedNodes );
        }
    }
}


void DAGNode::executeVisitorTopDown(simulation::Visitor* action, NodeList& executedNodes, StatusMap& statusMap, DAGNode* visitorRoot )
{
    if ( statusMap[this] != NOT_VISITED )
    {
        return; // skipped (already visited)
    }

    if( !this->isActive() )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;

        // in that case we can considerer if some child are activated, the graph is not valid, so no need to continue the recursion
        return;
    }

    if( this->isSleeping() && !action->canAccessSleepingNode )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;

        return;
    }

    // pour chaque noeud "prune" on continue à parcourir quand même juste pour marquer le noeud comme parcouru

    // check du "visitedStatus" des parents:
    // un enfant n'est pruné que si tous ses parents le sont
    // on ne passe à un enfant que si tous ses parents ont été visités
    bool allParentsPruned = true;
    bool hasParent = false;

    if( visitorRoot != this )
    {
        // the graph structure is generally modified during an action anterior to the traversal but can possibly be modified during the current traversal
        visitorRoot->updateDescendancy();

        const LinkParents::Container &parents = l_parents.getValue();
        for ( unsigned int i = 0; i < parents.size() ; i++ )
        {
            // if the visitor is run from a sub-graph containing a multinode linked with a node outside of the subgraph, do not consider the outside node by looking on the sub-graph descendancy
            if ( visitorRoot->_descendancy.contains(parents[i]) || parents[i]==visitorRoot )
            {
                // all parents must have been visited before
                if ( statusMap[parents[i]] == NOT_VISITED )
                    return; // skipped for now... the other parent should come later

                allParentsPruned = allParentsPruned && ( statusMap[parents[i]] == PRUNED );
                hasParent = true;
            }
        }
    }

    // all parents have been visited, let's go with the visitor
    if ( allParentsPruned && hasParent )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;

        // ... but continue the recursion anyway!
        if( action->childOrderReversed(this) )
            for(unsigned int i = unsigned(child.size()); i>0;)
                static_cast<DAGNode*>(child[--i].get())->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);
        else
            for(unsigned int i = 0; i<child.size(); ++i)
                static_cast<DAGNode*>(child[i].get())->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);
    }
    else
    {
        // execute the visitor on this node
        const Visitor::Result result = action->processNodeTopDown(this);

        // update status
        statusMap[this] = ( result == simulation::Visitor::RESULT_PRUNE ? PRUNED : VISITED );

        executedNodes.push_back(this);

        // ... and continue the recursion
        if( action->childOrderReversed(this) )
            for(unsigned int i = unsigned(child.size()); i>0;)
                static_cast<DAGNode*>(child[--i].get())->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);
        else
            for(unsigned int i = 0; i<child.size(); ++i)
                static_cast<DAGNode*>(child[i].get())->executeVisitorTopDown(action,executedNodes,statusMap,visitorRoot);

    }
}


// warning nodes that are dynamically created during the traversal, but that have not been traversed during the top-down, won't be traversed during the bottom-up
// TODO is it what we want?
// otherwise it is possible to restart from top, go to leaves and running bottom-up action while going up
void DAGNode::executeVisitorBottomUp( simulation::Visitor* action, NodeList& executedNodes )
{
    for( NodeList::reverse_iterator it = executedNodes.rbegin(), itend = executedNodes.rend() ; it != itend ; ++it )
    {
        (*it)->updateDescendancy();
        action->processNodeBottomUp( *it );
    }
}


void DAGNode::setDirtyDescendancy()
{
    _descendancy.clear();
    const LinkParents::Container &parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++ )
    {
        parents[i]->setDirtyDescendancy();
    }
}

void DAGNode::updateDescendancy()
{
    if( _descendancy.empty() && !child.empty() )
    {
        for(unsigned int i = 0; i<child.size(); ++i)
        {
            DAGNode* dagnode = static_cast<DAGNode*>(child[i].get());
            dagnode->updateDescendancy();
            _descendancy.insert( dagnode->_descendancy.begin(), dagnode->_descendancy.end() );
            _descendancy.insert( dagnode );
        }
    }
}



void DAGNode::executeVisitorTreeTraversal( simulation::Visitor* action, StatusMap& statusMap, Visitor::TreeTraversalRepetition repeat, bool alreadyRepeated )
{
    if( !this->isActive() )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;
        return;
    }

    if( this->isSleeping() && !action->canAccessSleepingNode )
    {
        // do not execute the visitor on this node
        statusMap[this] = PRUNED;
        return;
    }

    // node already visited and repetition must be avoid
    if( statusMap[this] != NOT_VISITED )
    {
        if( repeat==Visitor::NO_REPETITION || ( alreadyRepeated && repeat==Visitor::REPEAT_ONCE ) ) return;
        else alreadyRepeated = true;
    }

    if( action->processNodeTopDown(this) != simulation::Visitor::RESULT_PRUNE )
    {
        statusMap[this] = VISITED;
        if( action->childOrderReversed(this) )
            for(unsigned int i = unsigned(child.size()); i>0;)
                static_cast<DAGNode*>(child[--i].get())->executeVisitorTreeTraversal(action,statusMap,repeat,alreadyRepeated);
        else
            for(unsigned int i = 0; i<child.size(); ++i)
                static_cast<DAGNode*>(child[i].get())->executeVisitorTreeTraversal(action,statusMap,repeat,alreadyRepeated);
    }
    else
    {
        statusMap[this] = PRUNED;
    }

    action->processNodeBottomUp(this);
}


void DAGNode::initVisualContext()
{
    if (getNbParents())
    {
        this->setDisplayWorldGravity(false); //only display gravity for the root: it will be propagated at each time step
    }
}

void DAGNode::updateContext()
{
    sofa::core::objectmodel::BaseNode* firstParent = getFirstParent();

    if ( firstParent )
    {
        if( debug_ )
        {
            msg_info()<<"DAGNode::updateContext, node = "<<getName()<<", incoming context = "<< firstParent->getContext() ;
        }
        // TODO
        // ahem.... not sure here... which parent should I copy my context from exactly ?
        copyContext(*static_cast<Context*>(static_cast<DAGNode*>(firstParent)));
    }
    simulation::Node::updateContext();
}

void DAGNode::updateSimulationContext()
{
    sofa::core::objectmodel::BaseNode* firstParent = getFirstParent();

    if ( firstParent )
    {
        if( debug_ )
        {
            msg_info()<<"DAGNode::updateContext, node = "<<getName()<<", incoming context = "<< firstParent->getContext() ;
        }
        // TODO
        // ahem.... not sure here... which parent should I copy my simulation context from exactly ?
        copySimulationContext(*static_cast<Context*>(static_cast<DAGNode*>(firstParent)));
    }
    simulation::Node::updateSimulationContext();
}


Node* DAGNode::findCommonParent( simulation::Node* node2 )
{
    return static_cast<DAGNode*>(getRoot())->findCommonParent(this, static_cast<DAGNode*>(node2));
}

DAGNode* DAGNode::findCommonParent(DAGNode* node1, DAGNode* node2)
{
    updateDescendancy();

    if (!_descendancy.contains(node1) || !_descendancy.contains(node2))
        return nullptr; // this is NOT a parent

    // this is a parent
    for (unsigned int i = 0; i<child.size(); ++i)
    {
        // look for closer parents
        DAGNode* childcommon = static_cast<DAGNode*>(child[i].get())->findCommonParent(node1, node2);

        if (childcommon != nullptr)
            return childcommon;
    }
    // NO closer parents found
    return this;
}

void DAGNode::getLocalObjects( const sofa::core::objectmodel::ClassInfo& class_info, DAGNode::GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags ) const
{
    for (DAGNode::ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
    {
        sofa::core::objectmodel::BaseObject* obj = it->get();
        void* result = class_info.dynamicCast(obj);
        if (result != nullptr && (tags.empty() || (obj)->getTags().includes(tags)))
            container(result);
    }
}



//helper::Creator<xml::NodeElement::Factory, DAGNode> DAGNodeDefaultClass("default");
static helper::Creator<xml::NodeElement::Factory, DAGNode> DAGNodeClass("DAGNode");

} // namespace sofa::simulation::graph
