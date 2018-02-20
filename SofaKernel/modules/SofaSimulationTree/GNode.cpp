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
#include <SofaSimulationTree/GNode.h>
#include <sofa/simulation/Visitor.h>
#include <SofaSimulationCommon/xml/NodeElement.h>
#include <sofa/helper/Factory.inl>
#include <sofa/helper/cast.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

GNode::GNode(const std::string& name, GNode* parent)
    : simulation::Node(name)
    , l_parent(initLink("parent", "Parent node in the graph"))
{
    if( parent )
        parent->addChild((Node*)this);
}

GNode::~GNode()
{
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
    {
        GNode::SPtr gnode = sofa::core::objectmodel::SPtr_static_cast<GNode>(*it);
        gnode->l_parent.remove(this);
    }
}

/// Create, add, then return the new child of this Node
Node::SPtr GNode::createChild(const std::string& nodeName)
{
    GNode::SPtr newchild = sofa::core::objectmodel::New<GNode>(nodeName);
    this->addChild(newchild); newchild->updateSimulationContext();
    return newchild;
}

/// Add a child node
void GNode::doAddChild(GNode::SPtr node)
{
    child.add(node);
    node->l_parent.add(this);
}

/// Remove a child
void GNode::doRemoveChild(GNode::SPtr node)
{
    child.remove(node);
    node->l_parent.remove(this);
}


/// Add a child node
void GNode::addChild(core::objectmodel::BaseNode::SPtr node)
{
    GNode::SPtr gnode = down_cast<GNode>(node.get());
    notifyAddChild(gnode);
    doAddChild(gnode);
}

/// Remove a child
void GNode::removeChild(core::objectmodel::BaseNode::SPtr node)
{
    GNode::SPtr gnode = down_cast<GNode>(node.get());
    notifyRemoveChild(gnode);
    doRemoveChild(gnode);
}


/// Move a node from another node
void GNode::moveChild(BaseNode::SPtr node)
{
    GNode::SPtr gnode = down_cast<GNode>(node.get());
    if (!gnode) return;
    GNode* prev = gnode->parent();
    if (prev==NULL)
    {
        addChild(node);
    }
    else
    {
        notifyMoveChild(gnode,prev);
        prev->doRemoveChild(gnode);
        doAddChild(gnode);
    }
}


/// Remove a child
void GNode::detachFromGraph()
{
    GNode::SPtr me = this; // make sure we don't delete ourself before the end of this method
    if (parent())
        parent()->removeChild(this);
}

/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* GNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
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
        std::cout << "GNODE: search for object of type " << class_info.name() << std::endl;
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
                    std::cout << "GNODE: testing object " << (obj)->getName() << " of type " << (obj)->getClassName() << std::endl;
#endif
                result = class_info.dynamicCast(obj);
                if (result != NULL)
                {
#ifdef DEBUG_GETOBJECT
                    std::cout << "GNODE: found object " << (obj)->getName() << " of type " << (obj)->getClassName() << std::endl;
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
            dmsg_error("GNode") << "SearchRoot SHOULD NOT BE POSSIBLE HERE.";
            break;
        }
    }

    return result;
}

/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* GNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
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
            //dmsg_error("GNode") << "Child node "<<name<<" not found in "<<getPathName();
            return NULL;
        }
        else
        {
            core::objectmodel::BaseObject* obj = simulation::Node::getObject(name);
            if (obj == NULL)
            {
                //dmsg_error("GNode") << "Object "<<name<<" not found in "<<getPathName();
                return NULL;
            }
            else
            {
                void* result = class_info.dynamicCast(obj);
                if (result == NULL)
                {
                    dmsg_error("GNode") << "Object "<<name<<" in "<<getPathName()<<" does not implement class "<<class_info.name() ;
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
void GNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
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
            dmsg_error("GNode") << "SearchRoot SHOULD NOT BE POSSIBLE HERE.";
            break;
        }
    }
}

/// Get a list of parent node
core::objectmodel::BaseNode::Parents GNode::getParents() const
{
    Parents p;
    if (parent())
        p.push_back(parent());
    return p;
}

/// returns number of parents
size_t GNode::getNbParents() const
{
    return parent() ? 1 : 0;
}

/// return the first parent (returns NULL if no parent)
core::objectmodel::BaseNode* GNode::getFirstParent() const
{
    return parent();
}


/// Test if the given context is an ancestor of this context.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool GNode::hasAncestor(const BaseContext* context) const
{
    GNode* p = parent();
    while (p)
    {
        if (p==context) return true;
        p = p->parent();
    }
    return false;
}

/// Mesh Topology that is relevant for this context
/// (within it or its parents until a mapping is reached that does not preserve topologies).
core::topology::BaseMeshTopology* GNode::getActiveMeshTopology() const
{
    if (this->meshTopology)
        return this->meshTopology;
    // Check if a local mapping stops the search
    if (this->mechanicalMapping && !this->mechanicalMapping->sameTopology())
    {
        return NULL;
    }
    for ( Sequence<core::BaseMapping>::iterator i=this->mapping.begin(), iend=this->mapping.end(); i!=iend; ++i )
    {
        if (!(*i)->sameTopology())
        {
            return NULL;
        }
    }
    // No mapping with a different topology, continue on to the parent
    GNode* p = parent();
    if (!p)
    {
        return NULL;
    }
    else
    {
        return p->getActiveMeshTopology();
    }
}

/// Execute a recursive action starting from this node
void GNode::doExecuteVisitor(simulation::Visitor* action, bool)
{
#ifdef SOFA_DUMP_VISITOR_INFO
    action->setNode(this);
    action->printInfo(getContext(), true);
#endif

    if(action->processNodeTopDown(this) != simulation::Visitor::RESULT_PRUNE)
    {
        if (action->childOrderReversed(this))
        {
            for(unsigned int i = child.size(); i>0;)
            {
                child[--i]->executeVisitor(action);
            }
        }
        else
        {
            for(unsigned int i = 0; i<child.size(); ++i)
            {
                child[i]->executeVisitor(action);
            }
        }
    }

    action->processNodeBottomUp(this);

#ifdef SOFA_DUMP_VISITOR_INFO
    action->printInfo(getContext(), false);
#endif
}

void GNode::initVisualContext()
{
    if ( getNbParents() )
    {
        this->setDisplayWorldGravity(false); //only display gravity for the root: it will be propagated at each time step
    }
}

void GNode::updateContext()
{
    if ( getNbParents() )
    {
        if( debug_ )
        {
            msg_info()<<"GNode::updateContext, node = "<<getName()<<", incoming context = "<< *parent()->getContext() ;
        }
        copyContext(*parent());
    }
    simulation::Node::updateContext();
}

void GNode::updateSimulationContext()
{
    if ( getNbParents() )
    {
        if( debug_ )
        {
            msg_info()<<"GNode::updateContext, node = "<<getName()<<", incoming context = "<< *parent()->getContext() ;
        }
        copySimulationContext(*parent());
    }
    simulation::Node::updateSimulationContext();
}



Node* GNode::findCommonParent( simulation::Node* node2 )
{
    GNode *gnodeGroup1=this,
                             *gnodeGroup2=static_cast<GNode*>(node2);
    helper::vector<GNode*> hierarchyParent;

    gnodeGroup1=static_cast<GNode*>(gnodeGroup1->parent());
    while ( gnodeGroup1)
    {
        hierarchyParent.push_back(gnodeGroup1);
        gnodeGroup1=static_cast<GNode*>(gnodeGroup1->parent());
    }
    if (hierarchyParent.empty())   return NULL;

    gnodeGroup2=static_cast<GNode*>(gnodeGroup2->parent());
    while (gnodeGroup2)
    {
        helper::vector<GNode*>::iterator it=std::find(hierarchyParent.begin(), hierarchyParent.end(), gnodeGroup2);
        if (it != hierarchyParent.end())
        {
            return gnodeGroup2;
        }
        gnodeGroup2=static_cast<GNode*>(gnodeGroup2->parent());
    }

    return NULL;
}


SOFA_DECL_CLASS(GNode)

//helper::Creator<xml::NodeElement::Factory, GNode> GNodeDefaultClass("default");
helper::Creator<xml::NodeElement::Factory, GNode> GNodeClass("GNode");

} // namespace tree

} // namespace simulation

} // namespace sofa

