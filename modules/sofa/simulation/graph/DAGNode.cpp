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
    , l_parent(initLink("parent", "Parent node in the graph"))
{
    if( parent )
        parent->addChild((Node*)this);
}

DAGNode::~DAGNode()
{}

/// Create, add, then return the new child of this Node
Node* DAGNode::createChild(const std::string& nodeName)
{
    DAGNode* newchild = new DAGNode(nodeName);
    this->addChild(newchild); newchild->updateSimulationContext();
    return newchild;
}

/// Add a child node
void DAGNode::doAddChild(DAGNode::SPtr node)
{
    child.add(node);
    node->l_parent.add(this);
}

/// Remove a child
void DAGNode::doRemoveChild(DAGNode::SPtr node)
{
    child.remove(node);
    node->l_parent.remove(this);
}


/// Add a child node
void DAGNode::addChild(core::objectmodel::BaseNode::SPtr node)
{
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
    DAGNode* prev = dagnode->parent();
    if (prev==NULL)
    {
        addChild(node);
    }
    else
    {
        notifyMoveChild(dagnode,prev);
        prev->doRemoveChild(dagnode);
        doAddChild(dagnode);
    }
}


/// Remove a child
void DAGNode::detachFromGraph()
{
    DAGNode::SPtr me = this; // make sure we don't delete ourself before the end of this method
    if (parent())
        parent()->removeChild(this);
}

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* DAGNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    if (dir == SearchRoot)
    {
        if (parent() != NULL) return parent()->getObject(class_info, tags, dir);
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
            if (parent()) result = parent()->getObject(class_info, tags, SearchUp);
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
        return Node::getObject(class_info, Local);
    }
    else if (path[0] == '/')
    {
        if (parent()) return parent()->getObject(class_info, path);
        else return getObject(class_info,std::string(path,1));
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
        std::string newpath = std::string(path, 3);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        if (parent()) return parent()->getObject(class_info,newpath);
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
        if (parent() != NULL)
        {
            if (parent()->isActive())
            {
                parent()->getObjects(class_info, container, tags, dir);
                return;
            }
            else return;
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
            if (parent()) parent()->getObjects(class_info, container, tags, SearchUp);
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
    if (parent())
        p.push_back(parent());
    return p;
}

/// Get parent node (or NULL if no hierarchy or for root node)
core::objectmodel::BaseNode* DAGNode::getParent()
{
    return parent();
}

/// Get parent node (or NULL if no hierarchy or for root node)
const core::objectmodel::BaseNode* DAGNode::getParent() const
{
    return parent();
}

/// Test if the given context is an ancestor of this context.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool DAGNode::hasAncestor(const BaseContext* context) const
{
    DAGNode* p = parent();
    while (p)
    {
        if (p==context) return true;
        p = p->parent();
    }
    return false;
}

/// Execute a recursive action starting from this node
/// This method bypass the actionScheduler of this node if any.
void DAGNode::doExecuteVisitor(simulation::Visitor* action)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    action->setNode(this);
    action->printInfo(getContext(), true);
#endif
    if(action->processNodeTopDown(this) != simulation::Visitor::RESULT_PRUNE)
    {
        for(unsigned int i = 0; i<child.size(); ++i)
        {
            child[i]->executeVisitor(action);
        }
    }

    action->processNodeBottomUp(this);
#ifdef SOFA_DUMP_VISITOR_INFO
    action->printInfo(getContext(), false);
#endif
}


/// Return the full path name of this node
std::string DAGNode::getPathName() const
{
    std::string str;

    if (parent() != NULL)
    {
        str = parent()->getPathName();
        str += '/';
        str += getName();
    }

    return str;
}



void DAGNode::initVisualContext()
{
    if (getParent() != NULL)
    {
        this->setDisplayWorldGravity(false); //only display gravity for the root: it will be propagated at each time step
    }
}

void DAGNode::updateContext()
{
    if ( getParent() != NULL )
    {
        if( debug_ )
        {
            std::cerr<<"DAGNode::updateContext, node = "<<getName()<<", incoming context = "<< *getParent()->getContext() << std::endl;
        }
        copyContext(*parent());
    }
    simulation::Node::updateContext();
}

void DAGNode::updateSimulationContext()
{
    if ( getParent() != NULL )
    {
        if( debug_ )
        {
            std::cerr<<"DAGNode::updateContext, node = "<<getName()<<", incoming context = "<< *getParent()->getContext() << std::endl;
        }
        copySimulationContext(*parent());
    }
    simulation::Node::updateSimulationContext();
}

SOFA_DECL_CLASS(DAGNode)

//helper::Creator<xml::NodeElement::Factory, DAGNode> DAGNodeDefaultClass("default");
helper::Creator<xml::NodeElement::Factory, DAGNode> DAGNodeClass("DAGNode");

} // namespace graph

} // namespace simulation

} // namespace sofa

