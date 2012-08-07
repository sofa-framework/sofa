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
#include <sofa/simulation/graph/DAGNode.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/xml/NodeElement.h>
#include <sofa/helper/Factory.inl>

namespace sofa
{

namespace simulation
{

namespace graph
{

DAGNode::DAGNode(const std::string& name, DAGNode* parent)
    : simulation::Node(name)
    , l_parents(initLink("parents", "Parents nodes in the graph"))
{
    if( parent )
        parent->addChild((Node*)this);
}

DAGNode::~DAGNode()
{}

/// Create, add, then return the new child of this Node
Node::SPtr DAGNode::createChild(const std::string& nodeName)
{
    DAGNode::SPtr newchild = sofa::core::objectmodel::New<DAGNode>(nodeName);
    this->addChild(newchild); newchild->updateSimulationContext();
    return newchild;
}

/// Add a child node
void DAGNode::doAddChild(DAGNode::SPtr node)
{
//    printf("DAGNode::doAddChild this=%X(%s) child=%X(%s)\n",this,getName().c_str(),node.get(),node->getName().c_str());
    child.add(node);
    node->l_parents.add(this);
    node->l_parents.updateLinks(); // to fix load-time unresolved links
}

/// Remove a child
void DAGNode::doRemoveChild(DAGNode::SPtr node)
{
    child.remove(node);
    node->l_parents.remove(this);
}


/// Add a child node
void DAGNode::addChild(core::objectmodel::BaseNode::SPtr node)
{
//    printf("DAGNode::addChild this=%s child=%s\n",getName().c_str(),node->getName().c_str());
    DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_dynamic_cast<DAGNode>(node);
    notifyAddChild(dagnode);
    doAddChild(dagnode);
}

/// Remove a child
void DAGNode::removeChild(core::objectmodel::BaseNode::SPtr node)
{
    DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_dynamic_cast<DAGNode>(node);
    notifyRemoveChild(dagnode);
    doRemoveChild(dagnode);
}


/// Move a node from another node
void DAGNode::moveChild(BaseNode::SPtr node)
{
    DAGNode::SPtr dagnode = sofa::core::objectmodel::SPtr_dynamic_cast<DAGNode>(node);
    if (!dagnode) return;

    core::objectmodel::BaseNode::Parents  nodeParents = dagnode->getParents();
    if (nodeParents.empty())
    {
        addChild(node);
    }
    else
    {
        for (core::objectmodel::BaseNode::Parents::iterator it = nodeParents.begin(); it != nodeParents.end(); ++it)
        {
            DAGNode *prev = dynamic_cast<DAGNode*>(*it);
            notifyMoveChild(dagnode,prev);
            prev->doRemoveChild(dagnode);
        }
        doAddChild(dagnode);
    }
}


/// Remove a child
void DAGNode::detachFromGraph()
{
    DAGNode::SPtr me = this; // make sure we don't delete ourself before the end of this method
    //Sequence<BglNode>::iterator it=parents.begin(), it_end=parents.end();
    //for (;it!=it_end;++it) (*it)->removeChild(this);
    LinkParents::Container parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++)
    {
        parents[i]->removeChild(this);
    }
}

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* DAGNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if (dir == SearchRoot)
    {
        if (!getParents().empty()) return getRootContext()->getObject(class_info, tags, dir);
        else dir = SearchDown; // we are the root, search down from here.
    }
    void *result = NULL;
#ifdef DEBUG_GETOBJECT
    std::string cname = class_info.name();
    if (cname != std::string("N4sofa4core6ShaderE"))
        std::cout << "DAGNode: search for object of type " << class_info.name() << std::endl;
    std::string gname = "N4sofa9component8topology32TetrahedronSetGeometryAlgorithms";
    bool isg = cname.length() >= gname.length() && std::string(cname, 0, gname.length()) == gname;
#endif
    if (dir != SearchParents)
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            core::objectmodel::BaseObject* obj = it->get();
            if (tags.empty() || (obj)->getTags().includes(tags))
            {
#ifdef DEBUG_GETOBJECT
                if (isg)
                    std::cout << "DAGNode: testing object " << (obj)->getName() << " of type " << (obj)->getClassName() << std::endl;
#endif
                result = class_info.dynamicCast(obj);
                if (result != NULL)
                {
#ifdef DEBUG_GETOBJECT
                    std::cout << "DAGNode: found object " << (obj)->getName() << " of type " << (obj)->getClassName() << std::endl;
#endif
                    break;
                }
            }
        }

    if (result == NULL)
    {
        switch(dir)
        {
        case Local:
            break;
        case SearchParents:
        case SearchUp:
        {
            Parents parents = getParents();
            if (!parents.empty())
                for (Parents::iterator it = parents.begin(); it!=parents.end() && !result; it++)
                    result = dynamic_cast<Node*>(*it)->getObject(class_info, tags, SearchUp);
        }
        break;
        case SearchDown:
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                result = (*it)->getObject(class_info, tags, dir);
                if (result != NULL) break;
            }
            break;
        case SearchRoot:
            std::cerr << "SearchRoot SHOULD NOT BE POSSIBLE HERE!\n";
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
        Parents parents = getParents();
        if (parents.empty()) return getObject(class_info,std::string(path,1));
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
        Parents parents = getParents();
        if (!parents.empty())
        {
            for (Parents::iterator it = parents.begin(); it!=parents.end(); it++)
            {
                void* obj = dynamic_cast<Node*>(*it)->getObject(class_info,newpath);
                if (obj) return obj;
            }
            return 0;   // not found in any parent node at all
        }
        else return getObject(class_info,newpath);
    }
    else
    {
        std::string::size_type pend = path.find('/');
        if (pend == std::string::npos) pend = path.length();
        std::string name ( path, 0, pend );
        Node* child = getChild(name);
        if (child)
        {
            while (pend < path.length() && path[pend] == '/')
                ++pend;
            return child->getObject(class_info, std::string(path, pend));
        }
        else if (pend < path.length())
        {
            //std::cerr << "ERROR: child node "<<name<<" not found in "<<getPathName()<<std::endl;
            return NULL;
        }
        else
        {
            core::objectmodel::BaseObject* obj = simulation::Node::getObject(name);
            if (obj == NULL)
            {
                //std::cerr << "ERROR: object "<<name<<" not found in "<<getPathName()<<std::endl;
                return NULL;
            }
            else
            {
                void* result = class_info.dynamicCast(obj);
                if (result == NULL)
                {
                    std::cerr << "ERROR: object "<<name<<" in "<<getPathName()<<" does not implement class "<<class_info.name()<<std::endl;
                    return NULL;
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
    if (dir == SearchRoot)
    {
        if (!getParents().empty())
        {
            getRootContext()->getObjects(class_info, container, tags, dir);
            return;
        }
        else dir = SearchDown; // we are the root, search down from here.
    }
    if (dir != SearchParents)
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            core::objectmodel::BaseObject* obj = it->get();
            void* result = class_info.dynamicCast(obj);
            if (result != NULL && (tags.empty() || (obj)->getTags().includes(tags)))
                container(result);
        }

    {
        switch(dir)
        {
        case Local:
            break;
        case SearchParents:
        case SearchUp:
        {
            // WORK IN PROGRESS
            // TODO: manage diamond setups to avoid multiple getObjects() calls on a Node...
            Parents parents = getParents();
            for (Parents::iterator it = parents.begin(); it!=parents.end(); it++)
                dynamic_cast<Node*>(*it)->getObjects(class_info, container, tags, SearchUp);
        }
        break;
        case SearchDown:
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                if ((*it)->isActive())
                    (*it)->getObjects(class_info, container, tags, dir);
            }
            break;
        case SearchRoot:
            std::cerr << "SearchRoot SHOULD NOT BE POSSIBLE HERE!\n";
            break;
        }
    }
}

/// Get a list of parent node
core::objectmodel::BaseNode::Parents DAGNode::getParents() const
{
    Parents p;

    LinkParents::Container parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++)
    {
        if (parents[i])
        {
            p.push_back(parents[i]);
//            printf("DAGNode::getParents() \"%s\"=%X parents[%d]=%X\"%s\"\n",getName().c_str(),this,i,(void*)parents[i],parents[i]->getName().c_str());
        }
//        else
//            printf("DAGNode::getParents() \"%s\"=%X parents[%d]=%X\n",getName().c_str(),this,i,(void*)parents[i]);
    }

    return p;
}

/// Test if the given node is a parent of this node.
bool DAGNode::hasParent(const BaseNode* node) const
{
    Parents p = getParents();
    return (p.end() != std::find(p.begin(), p.end(), node));
}

/// Test if the given context is a parent of this context.
bool DAGNode::hasParent(const BaseContext* context) const
{
    if (context == NULL) return getParents().empty();

    LinkParents::Container parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++)
        if (context == parents[i]->getContext()) return true;
    return false;

}



/// Test if the given context is an ancestor of this context.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool DAGNode::hasAncestor(const BaseContext* context) const
{
    LinkParents::Container parents = l_parents.getValue();
    for ( unsigned int i = 0; i < parents.size() ; i++)
        if (context == parents[i]->getContext()
            || parents[i]->hasAncestor(context))
            return true;
    return false;
}

/// Execute a recursive action starting from this node
/// This method bypass the actionScheduler of this node if any.
void DAGNode::doExecuteVisitor(simulation::Visitor* action)
{
    // TODO
    // construire un sous-graphe de parcours du graphe et un "itérateur" pour cette liste
    // pour chaque noeud "prune" on continue à parcourir quand même juste pour marquer le noeud comme parcouru
    // on ne passe à un enfant que si tous ses parents ont été visités
    // un enfant n'est pruné que si tous ses parents le sont
    // NE PAS stocker les infos de parcours dans le DAGNode, plusieurs visiteurs pouvant parcourir le graphe simultanément

    DAGSubGraphNode* subGraph = createSubGraphDownward(NULL);
    DAGSubGraphNode::Nodes visitedNodes;
    subGraph->executeVisitorTopDown(action,&visitedNodes);
    while (!visitedNodes.empty())
    {
        visitedNodes.back()->executeVisitorBottomUp(action);
        visitedNodes.pop_back();
    }

    delete subGraph;
}


/// Return the full path name of this node
std::string DAGNode::getPathName() const
{
    std::string str;
    Parents parents = getParents();
    if (!parents.empty())
    {
        // for the full path name, we arbitrarily take the first parent of the list...
        // no smarter choice without breaking the "Node" heritage
        str = parents[0]->getPathName();
        str += '/';
        str += getName();
    }

    return str;
}



void DAGNode::initVisualContext()
{
    if (!getParents().empty())
    {
        this->setDisplayWorldGravity(false); //only display gravity for the root: it will be propagated at each time step
    }
}

void DAGNode::updateContext()
{
    if ( !getParents().empty() )
    {
        if( debug_ )
        {
            std::cerr<<"DAGNode::updateContext, node = "<<getName()<<", incoming context = "<< getParents()[0]->getContext() << std::endl;
        }
        // TODO
        // ahem.... not sure here... which parent should I copy my context from exactly ?
        copyContext(*dynamic_cast<Context*>(getParents()[0]));
    }
    simulation::Node::updateContext();
}

void DAGNode::updateSimulationContext()
{
    if ( !getParents().empty() )
    {
        if( debug_ )
        {
            std::cerr<<"DAGNode::updateContext, node = "<<getName()<<", incoming context = "<< getParents()[0]->getContext() << std::endl;
        }
        // TODO
        // ahem.... not sure here... which parent should I copy my simulation context from exactly ?
        copySimulationContext(*dynamic_cast<Context*>(getParents()[0]));
    }
    simulation::Node::updateSimulationContext();
}

/// build a subgraph of DAGSubGraphNode objects
DAGSubGraphNode* DAGNode::createSubGraphDownward(DAGSubGraphNode *parent)
{
//    std::cout << "DAGNode::createSubGraphDownward " << getName() << std::endl;

    DAGSubGraphNode *node = new DAGSubGraphNode(this);
    if (parent) parent->addChild(node);
    for (ChildIterator it = child.begin(); it!=child.end(); it++)
    {
        DAGSubGraphNode *childSubGraphNode = node->getRoot()->findNode(dynamic_cast<DAGNode*>(it->get()),DAGSubGraphNode::downward);
        // create child, or multi-map it ?
        if (!childSubGraphNode)
            dynamic_cast<DAGNode*>(it->get())->createSubGraphDownward(node);
        else
            node->addChild(childSubGraphNode);
    }
    return node;

}



SOFA_DECL_CLASS(DAGNode)

//helper::Creator<xml::NodeElement::Factory, DAGNode> DAGNodeDefaultClass("default");
helper::Creator<xml::NodeElement::Factory, DAGNode> DAGNodeClass("DAGNode");

} // namespace graph

} // namespace simulation

} // namespace sofa

