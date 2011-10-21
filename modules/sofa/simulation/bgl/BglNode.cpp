/******************************************************************************
 *       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
 *                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
 *                                                                             *
 * This program is free software; you can redistribute it and/or modify it     *
 * under the terms of the GNU General Public License as published by the Free  *
 * Software Foundation; either version 2 of the License, or (at your option)   *
 * any later version.                                                          *
 *                                                                             *
 * This program is distributed in the hope that it will be useful, but WITHOUT *
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
 * more details.                                                               *
 *                                                                             *
 * You should have received a copy of the GNU General Public License along     *
 * with this program; if not, write to the Free Software Foundation, Inc., 51  *
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
 *******************************************************************************
 *                            SOFA :: Applications                             *
 *                                                                             *
 * Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
 * H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
 * M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
 *                                                                             *
 * Contact information: contact@sofa-framework.org                             *
 ******************************************************************************/
//
// C++ Implementation: BglNode
//
// Description:
//
//
// Author: Francois Faure in The SOFA team </www.sofa-framework.org>, (C) 2008
//
// Copyright: See COPYING file that comes with this distribution
//
//
#include <sofa/simulation/bgl/BglNode.h>
#include <sofa/simulation/bgl/GetObjectsVisitor.h>
#include <sofa/simulation/bgl/BglGraphManager.inl>
#include <sofa/simulation/common/xml/NodeElement.h>

//Components of the core to detect during the addition of objects in a node
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/core/behavior/BaseInteractionConstraint.h>
#include <sofa/core/behavior/BaseInteractionProjectiveConstraintSet.h>

#include <boost/version.hpp>

#if BOOST_VERSION < 104200
#include <boost/vector_property_map.hpp>
#else
#include <boost/property_map/vector_property_map.hpp>
#endif
#include <sofa/simulation/bgl/dfv_adapter.h>
#include <sofa/helper/Factory.inl>

#define BREADTH_FIRST_VISIT

namespace sofa
{
namespace simulation
{
namespace bgl
{

unsigned int BglNode::uniqueId=0;
std::deque<unsigned int> BglNode::freeId;

BglNode::BglNode(const std::string& name)
    : sofa::simulation::Node(name)
{
    id=getUniqueId();
    BglGraphManager::getInstance()->addVertex(this);
}

BglNode::~BglNode()
{
    BglGraphManager::getInstance()->removeVertex(this);
    freeId.push_back(id);
}

unsigned int BglNode::getUniqueId()
{
    if (freeId.empty())
        return uniqueId++;
    else
    {
        int unique=freeId.front();
        freeId.pop_front();
        return unique;
    }
}


bool BglNode::addObject(core::objectmodel::BaseObject::SPtr sobj)
{
    using sofa::core::objectmodel::Tag;
    core::objectmodel::BaseObject* obj = sobj.get();
    sofa::core::BaseMapping* mm = dynamic_cast<sofa::core::BaseMapping*>(obj);
    if (mm && mm->isMechanical() )
    {
        sofa::core::behavior::BaseMechanicalState
        *msFrom=mm->getMechFrom()[0],
         *msTo  =mm->getMechTo()[0];

        if (msFrom && msTo)
        {
            Node *from=(Node*)msFrom->getContext();
            Node *to=(Node*)  msTo  ->getContext();
            BglGraphManager::getInstance()->addInteraction( from, to, mm);
        }
    }
    else if (sofa::core::behavior::BaseInteractionForceField* iff = dynamic_cast<sofa::core::behavior::BaseInteractionForceField*>(obj))
    {
        sofa::core::behavior::BaseMechanicalState
        *ms1=iff->getMechModel1(),
         *ms2=iff->getMechModel2();

        if (ms1 && ms2)
        {
            Node *m1=(Node*)ms1->getContext();
            Node *m2=(Node*)ms2->getContext();
            if (m1!=m2) BglGraphManager::getInstance()->addInteraction( m1, m2, iff);
        }
    }
    else if (sofa::core::behavior::BaseInteractionProjectiveConstraintSet* ic = dynamic_cast<sofa::core::behavior::BaseInteractionProjectiveConstraintSet*>(obj))
    {
        sofa::core::behavior::BaseMechanicalState
        *ms1=ic->getMechModel1(),
         *ms2=ic->getMechModel2();

        if (ms1 && ms2)
        {
            Node *m1=(Node*)ms1->getContext();
            Node *m2=(Node*)ms2->getContext();
            if (m1!=m2) BglGraphManager::getInstance()->addInteraction( m1, m2, ic);
        }
    }
    else if (sofa::core::behavior::BaseInteractionConstraint* ic = dynamic_cast<sofa::core::behavior::BaseInteractionConstraint*>(obj))
    {
        sofa::core::behavior::BaseMechanicalState
        *ms1=ic->getMechModel1(),
         *ms2=ic->getMechModel2();

        if (ms1 && ms2)
        {
            Node *m1=(Node*)ms1->getContext();
            Node *m2=(Node*)ms2->getContext();
            if (m1!=m2) BglGraphManager::getInstance()->addInteraction( m1, m2, ic);
        }
    }
    return Node::addObject(sobj);
}

bool BglNode::removeObject(core::objectmodel::BaseObject::SPtr sobj)
{
    core::objectmodel::BaseObject* obj = sobj.get();
    if (sofa::core::BaseMapping* mm = dynamic_cast<sofa::core::BaseMapping*>(obj))
    {
        if(mm->isMechanical())
            BglGraphManager::getInstance()->removeInteraction(mm);
    }
    else if (sofa::core::behavior::BaseInteractionForceField* iff = dynamic_cast<sofa::core::behavior::BaseInteractionForceField*>(obj))
    {
        BglGraphManager::getInstance()->removeInteraction(iff);
    }
    else if (sofa::core::behavior::BaseInteractionConstraint* ic = dynamic_cast<sofa::core::behavior::BaseInteractionConstraint*>(obj))
    {
        BglGraphManager::getInstance()->removeInteraction(ic);
    }
    return Node::removeObject(sobj);
}

void BglNode::addParent(BglNode *node)
{
    parents.add(node);
}
void BglNode::removeParent(BglNode *node)
{
    parents.remove(node);
}

/// Create, add, then return the new child of this Node
Node* BglNode::createChild(const std::string& nodeName)
{
    BglNode* newchild = new BglNode(nodeName);
    this->addChild(newchild); newchild->updateSimulationContext();
    return newchild;
}

void BglNode::addChild(core::objectmodel::BaseNode::SPtr c)
{
    BglNode::SPtr childNode = sofa::core::objectmodel::SPtr_static_cast< BglNode >(c);

    notifyAddChild(childNode);
    doAddChild(childNode);
}
/// Add a child node
void BglNode::doAddChild(BglNode::SPtr node)
{
    child.add(node);
    node->addParent(this);
    BglGraphManager::getInstance()->addEdge(this,node.get());
}


/// Remove a child
void BglNode::removeChild(core::objectmodel::BaseNode::SPtr c)
{
    BglNode::SPtr childNode = sofa::core::objectmodel::SPtr_static_cast< BglNode >(c);
    notifyRemoveChild(childNode);
    doRemoveChild(childNode);
}

void BglNode::doRemoveChild(BglNode::SPtr node)
{
    child.remove(node);
    node->removeParent(this);
    BglGraphManager::getInstance()->removeEdge(this, node.get());
}


void BglNode::moveChild(core::objectmodel::BaseNode::SPtr node)
{
    BglNode::SPtr childNode = sofa::core::objectmodel::SPtr_static_cast< BglNode >(node);
    if (!childNode) return;

    typedef std::vector< BglNode*> ParentsContainer;
    ParentsContainer nodeParents; childNode->getParents(nodeParents);
    if (nodeParents.empty())
    {
        addChild(node);
    }
    else
    {
        for (ParentsContainer::iterator it = nodeParents.begin(); it != nodeParents.end(); ++it)
        {
            BglNode *prev = (*it);
            notifyMoveChild(childNode,prev);
            prev->doRemoveChild(childNode);
        }
        doAddChild(childNode);
    }
}

void BglNode::detachFromGraph()
{
    //Sequence<BglNode>::iterator it=parents.begin(), it_end=parents.end();
    //for (;it!=it_end;++it) (*it)->removeChild(this);
    for ( unsigned int i = 0; i < parents.size() ; i++)
    {
        parents[i]->removeChild(this);
    }
}

/// Get a list of parent node
core::objectmodel::BaseNode::Parents BglNode::getParents() const
{
    core::objectmodel::BaseNode::Parents p;
    getParents(p);
    return p;
}

/// Test if the given context is a parent of this context.
bool BglNode::hasParent(const BaseContext* context) const
{
    if (context == NULL) return parents.empty();
    for (Sequence<BglNode>::iterator it=parents.begin(), it_end=parents.end(); it!=it_end; ++it)
    {
        BglNode* p = *it;
        if (p==context) return true;
    }
    return false;
}

/// Test if the given context is an ancestor of this context.
/// An ancestor is a parent or (recursively) the parent of an ancestor.
bool BglNode::hasAncestor(const BaseContext* context) const
{
    for (Sequence<BglNode>::iterator it=parents.begin(), it_end=parents.end(); it!=it_end; ++it)
    {
        BglNode* p = *it;
        if (p==context) return true;
        if (p->hasAncestor(context)) return true;
    }
    return false;
}


std::string BglNode::getPathName() const
{
    std::string str;
    if (!parents.empty())
        str = (*parents.begin())->getPathName();
    str += '/';
    str += getName();
    return str;

}


void BglNode::doExecuteVisitor( Visitor* visit )
{
#ifdef BREADTH_FIRST_VISIT
    BglGraphManager::getInstance()->breadthFirstVisit(this, *visit, SearchDown);
#else
    BglGraphManager::getInstance()->depthFirstVisit(this, *visit, SearchDown);
#endif
}



/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BglNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    if (dir == Local)
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            core::objectmodel::BaseObject* obj = it->get();
            void* result = class_info.dynamicCast(obj);
            if (result != NULL && (tags.empty() || (obj)->getTags().includes(tags)))
            {
                return result;
            }
        }
        return NULL;
    }
    else
    {
        GetObjectVisitor getobj(params /* PARAMS FIRST */, class_info, (dir == SearchParents ? this : NULL));
        getobj.setTags(tags);
#ifdef BREADTH_FIRST_VISIT
        BglGraphManager::getInstance()->breadthFirstVisit(this, getobj, dir);
#else
        BglGraphManager::getInstance()->depthFirstVisit(this, getobj, dir);
#endif
        return getobj.getObject();
    }
}


/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void BglNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, const sofa::core::objectmodel::TagSet& tags, SearchDirection dir) const
{
    sofa::core::ExecParams* params = sofa::core::ExecParams::defaultInstance();
    if (dir == Local)
    {
        for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
        {
            core::objectmodel::BaseObject* obj = it->get();
            void* result = class_info.dynamicCast(obj);
            if (result != NULL && (tags.empty() || (obj)->getTags().includes(tags)))
                container(result);
        }
    }
    else
    {
        GetObjectsVisitor getobjs(params /* PARAMS FIRST */, class_info, container, (dir == SearchParents ? this : NULL));
        getobjs.setTags(tags);
#ifdef BREADTH_FIRST_VISIT
        BglGraphManager::getInstance()->breadthFirstVisit(this, getobjs, dir);
#else
        BglGraphManager::getInstance()->depthFirstVisit(this, getobjs, dir);
#endif
    }
}

/// Generic object access, given a path from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* BglNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
{

    if (path.empty())
    {
        return Node::getObject(class_info, Local);
    }
    else if (path[0] == '/')
    {
        if (!parents.empty())
        {
            for (Parents::iterator it=parents.begin(); it!=parents.end(); ++it)
            {
                void *result=(*it)->getObject(class_info, path);
                if (result) return result;
            }
            return NULL;
        }
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

        if (!parents.empty())
        {
            for (Parents::iterator it=parents.begin(); it!=parents.end(); ++it)
            {
                void *result=(*it)->getObject(class_info, newpath);
                if (result) return result;
            }
            return NULL;
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
            std::cerr << "ERROR: child node "<<name<<" not found in "<<getPathName()<<std::endl;
            return NULL;
        }
        else
        {
            core::objectmodel::BaseObject* obj = simulation::Node::getObject(name);
            if (obj == NULL)
            {
                std::cerr << "ERROR: object "<<name<<" not found in "<<getPathName()<<std::endl;
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



void BglNode::updateContext()
{
    if (!parents.empty())
    {
        copyContext(*(*parents.begin()));
    }
    simulation::Node::updateContext();
}

void BglNode::updateSimulationContext()
{
    if (!parents.empty())
    {
        copySimulationContext(*(*parents.begin()));
    }
    simulation::Node::updateSimulationContext();
}


void BglNode::initVisualContext()
{
    if (!parents.empty())
    {
        this->setDisplayWorldGravity(false); //only display gravity for the root: it will be propagated at each time step
    }
}

SOFA_DECL_CLASS(BglNode)

helper::Creator<simulation::xml::NodeElement::Factory, BglNode> BglNodeClass("BglNode");

} // namespace bgl

} // namespace simulation

} // namespace sofa
