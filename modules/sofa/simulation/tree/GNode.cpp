/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/simulation/tree/GNode.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/common/DesactivatedNodeVisitor.h>
#include <sofa/simulation/tree/MutationListener.h>
#include <sofa/simulation/tree/xml/NodeElement.h>
#include <iostream>

using std::cerr;
using std::endl;

namespace sofa
{

namespace simulation
{

namespace tree
{

using helper::system::thread::CTime;
using namespace sofa::core::objectmodel;

GNode::GNode(const std::string& name, GNode* parent)
    : simulation::Node(name)
{
    if( parent )
        parent->addChild(this);
}

GNode::~GNode()
{}

/// Add a child node
void GNode::doAddChild(GNode* node)
{
    child.add(node);
    node->parent.add(this);
}

/// Remove a child
void GNode::doRemoveChild(GNode* node)
{
    child.remove(node);
    node->parent.remove(this);
}

/// Add a child node
void GNode::addChild(GNode* node)
{
    notifyAddChild(node);
    doAddChild(node);
}

/// Remove a child
void GNode::removeChild(GNode* node)
{
    notifyRemoveChild(node);
    doRemoveChild(node);
}

/// Add a child node
void GNode::addChild(core::objectmodel::BaseNode* node)
{
    this->addChild(dynamic_cast<GNode*>(node));
}

/// Remove a child node
void GNode::removeChild(core::objectmodel::BaseNode* node)
{
    this->removeChild(dynamic_cast<GNode*>(node));
}

/// Move a node from another node
void GNode::moveChild(GNode* node)
{
    GNode* prev = node->parent;
    if (prev==NULL)
    {
        addChild(node);
    }
    else
    {
        notifyMoveChild(node,prev);
        prev->doRemoveChild(node);
        doAddChild(node);
    }
}



/// Generic object access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void* GNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, SearchDirection dir) const
{
    if (dir == SearchRoot)
    {
        if (parent != NULL) return parent->getObject(class_info, dir);
        else dir = SearchDown; // we are the root, search down from here.
    }
    void *result = NULL;
    for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
    {
        result = class_info.dynamicCast(*it);
        if (result != NULL) break;
    }
    if (result == NULL)
    {
        switch(dir)
        {
        case Local:
            break;
        case SearchUp:
            if (parent) result = parent->getObject(class_info, dir);
            break;
        case SearchDown:
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                result = (*it)->getObject(class_info, dir);
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
void* GNode::getObject(const sofa::core::objectmodel::ClassInfo& class_info, const std::string& path) const
{
    if (path.empty())
    {
        return getObject(class_info, Local);
    }
    else if (path[0] == '/')
    {
        if (parent) return parent->getObject(class_info, path);
        else return getObject(class_info,std::string(path,1));
    }
    else if (std::string(path,0,2)==std::string("./"))
    {
        std::string newpath = std::string(path, 2);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        return simulation::Node::getObject(newpath);
    }
    else if (std::string(path,0,3)==std::string("../"))
    {
        std::string newpath = std::string(path, 3);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        if (parent) return parent->simulation::Node::getObject(newpath);
        else return simulation::Node::getObject(newpath);
    }
    else
    {
        std::string::size_type pend = path.find('/');
        if (pend == std::string::npos) pend = path.length();
        std::string name ( path, 0, pend );
        GNode* child = getChild(name);
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


/// Generic list of objects access, possibly searching up or down from the current context
///
/// Note that the template wrapper method should generally be used to have the correct return type,
void GNode::getObjects(const sofa::core::objectmodel::ClassInfo& class_info, GetObjectsCallBack& container, SearchDirection dir) const
{
    if (dir == SearchRoot)
    {
        if (parent != NULL)
        {
            if (parent->isActive())
            {
                parent->getObjects(class_info, container, dir);
                return;
            }
            else return;
        }
        else dir = SearchDown; // we are the root, search down from here.
    }
    for (ObjectIterator it = this->object.begin(); it != this->object.end(); ++it)
    {
        void* result = class_info.dynamicCast(*it);
        if (result != NULL)
            container(result);
    }

    {
        switch(dir)
        {
        case Local:
            break;
        case SearchUp:
            if (parent) parent->getObjects(class_info, container, dir);
            break;
        case SearchDown:
            for(ChildIterator it = child.begin(); it != child.end() && (*it)->isActive(); ++it)
            {
                (*it)->getObjects(class_info, container, dir);
            }
            break;
        case SearchRoot:
            std::cerr << "SearchRoot SHOULD NOT BE POSSIBLE HERE!\n";
            break;
        }
    }
}




/// Get parent node (or NULL if no hierarchy or for root node)
core::objectmodel::BaseNode* GNode::getParent()
{
    return parent;
}

/// Get parent node (or NULL if no hierarchy or for root node)
const core::objectmodel::BaseNode* GNode::getParent() const
{
    return parent;
}

/// Get parent node (or NULL if no hierarchy or for root node)
sofa::helper::vector< core::objectmodel::BaseNode* > GNode::getChildren()
{
    sofa::helper::vector< core::objectmodel::BaseNode* > list_children;
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
    {
        list_children.push_back((*it));
    }
    return list_children;
}

/// Get parent node (or NULL if no hierarchy or for root node)
const sofa::helper::vector< core::objectmodel::BaseNode* > GNode::getChildren() const
{
    sofa::helper::vector< core::objectmodel::BaseNode* > list_children;
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
    {
        list_children.push_back((*it));
    }
    return list_children;
}



/// Execute a recursive action starting from this node
/// This method bypass the actionScheduler of this node if any.
void GNode::doExecuteVisitor(simulation::Visitor* action)
{
    if (getLogTime())
    {
        const ctime_t t0 = CTime::getTime();
        ctime_t tChild = 0;
        actionStack.push(action);
        if(action->processNodeTopDown(this) != simulation::Visitor::RESULT_PRUNE)
        {
            ctime_t ct0 = CTime::getTime();
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                (*it)->executeVisitor(action);
            }
            tChild = CTime::getTime() - ct0;
        }
        action->processNodeBottomUp(this);
        actionStack.pop();
        ctime_t tTree = CTime::getTime() - t0;
        ctime_t tNode = tTree - tChild;
        if (actionStack.empty())
        {
            totalTime.tNode += tNode;
            totalTime.tTree += tTree;
            ++totalTime.nVisit;
        }
        NodeTimer& t = actionTime[action->getCategoryName()];
        t.tNode += tNode;
        t.tTree += tTree;
        ++t.nVisit;
        if (!actionStack.empty())
        {
            // remove time from calling action log
            simulation::Visitor* prev = actionStack.top();
            NodeTimer& t = actionTime[prev->getCategoryName()];
            t.tNode -= tTree;
            t.tTree -= tTree;
        }
    }
    else
    {
        if(action->processNodeTopDown(this) != simulation::Visitor::RESULT_PRUNE)
        {
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                (*it)->executeVisitor(action);
            }
        }
        action->processNodeBottomUp(this);
    }
}


/// Find a child node given its name
GNode* GNode::getChild(const std::string& name) const
{
    for (ChildIterator it = child.begin(), itend = child.end(); it != itend; ++it)
        if ((*it)->getName() == name)
            return *it;
    return NULL;
}

/// Get a descendant node given its name
GNode* GNode::getTreeNode(const std::string& name) const
{
    GNode* result = NULL;
    result = getChild(name);
    for (ChildIterator it = child.begin(), itend = child.end(); result == NULL && it != itend; ++it)
        result = (*it)->getTreeNode(name);
    return result;
}

void GNode::notifyAddChild(GNode* node)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->addChild(this, node);
}

void GNode::notifyRemoveChild(GNode* node)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->removeChild(this, node);
}

void GNode::notifyMoveChild(GNode* node, GNode* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveChild(prev, this, node);
}

/// Return the full path name of this node
std::string GNode::getPathName() const
{
    std::string str;
    if (parent!=NULL) str = parent->getPathName();
    str += '/';
    str += getName();
    return str;
}

/// Topology
core::componentmodel::topology::Topology* GNode::getTopology() const
{
    // return this->topology;
    // CHANGE 12/01/06 (Jeremie A.): Inherit parent topology if no local topology is defined
    if (this->topology)
        return this->topology;
    else if (parent)
        return parent->getTopology();
    else
        return NULL;
}

/// Dynamic Topology
core::componentmodel::topology::BaseTopology* GNode::getMainTopology() const
{
    core::componentmodel::topology::BaseTopology *main=0;
    unsigned int i;
    for (i=0; i<basicTopology.size(); ++i)
    {
        if (basicTopology[i]->isMainTopology()==true)
            main=basicTopology[i];
    }
    // return main;
    // CHANGE 12/01/06 (Jeremie A.): Inherit parent topology if no local topology is defined
    if (main)
        return main;
    else if (parent)
        return parent->getMainTopology();
    else
        return NULL;
}

/// Mesh Topology (unified interface for both static and dynamic topologies)
core::componentmodel::topology::BaseMeshTopology* GNode::getMeshTopology() const
{
    if (this->meshTopology)
        return this->meshTopology;
    else if (parent)
        return parent->getMeshTopology();
    else
        return NULL;
}

/// Shader
core::objectmodel::BaseObject* GNode::getShader() const
{
    if (shader)
        return shader;
    else if (parent)
        return parent->getShader();
    else
        return NULL;
}

/// Mechanical Degrees-of-Freedom
core::objectmodel::BaseObject* GNode::getMechanicalState() const
{
    // return this->mechanicalModel;
    // CHANGE 12/01/06 (Jeremie A.): Inherit parent mechanical model if no local model is defined
    if (this->mechanicalState)
        return this->mechanicalState;
    else if (parent)
        return parent->getMechanicalState();
    else
        return NULL;
}

/// Update the parameters of the System
void GNode::reinit()
{
    sofa::simulation::DesactivationVisitor desactivate(isActive());
    desactivate.execute( this );
}


void GNode::initVisualContext()
{
    if (getParent() != NULL)
    {
        if (showVisualModels_.getValue() == -1)
            showVisualModels_.setValue(static_cast<GNode *>(getParent())->showVisualModels_.getValue());
        if (showBehaviorModels_.getValue() == -1)
            showBehaviorModels_.setValue(static_cast<GNode *>(getParent())->showBehaviorModels_.getValue());
        if (showCollisionModels_.getValue() == -1)
            showCollisionModels_.setValue(static_cast<GNode *>(getParent())->showCollisionModels_.getValue());
        if (showBoundingCollisionModels_.getValue() == -1)
            showBoundingCollisionModels_.setValue(static_cast<GNode *>(getParent())->showBoundingCollisionModels_.getValue());
        if (showMappings_.getValue() == -1)
            showMappings_.setValue(static_cast<GNode *>(getParent())->showMappings_.getValue());
        if (showMechanicalMappings_.getValue() == -1)
            showMechanicalMappings_.setValue(static_cast<GNode *>(getParent())->showMechanicalMappings_.getValue());
        if (showForceFields_.getValue() == -1)
            showForceFields_.setValue(static_cast<GNode *>(getParent())->showForceFields_.getValue());
        if (showInteractionForceFields_.getValue() == -1)
            showInteractionForceFields_.setValue(static_cast<GNode *>(getParent())->showInteractionForceFields_.getValue());
        if (showWireFrame_.getValue() == -1)
            showWireFrame_.setValue(static_cast<GNode *>(getParent())->showWireFrame_.getValue());
        if (showNormals_.getValue() == -1)
            showNormals_.setValue(static_cast<GNode *>(getParent())->showNormals_.getValue());
    }
}

void GNode::updateContext()
{
    if ( getParent() != NULL )
    {
        copyContext(*parent);
        //cerr<<"node "<<getName()<<", copy context, time = "<<getTime()<<endl;
    }
    simulation::Node::updateContext();
}

void GNode::updateSimulationContext()
{
    if ( getParent() != NULL )
    {
        copySimulationContext(*parent);
    }
    simulation::Node::updateSimulationContext();
}

void GNode::updateVisualContext(int FILTER)
{
    if ( getParent() != NULL )
    {
        if (!FILTER)
            copyVisualContext(*parent);
        else
        {
            switch (FILTER)
            {
            case 1:
                showVisualModels_.setValue((*parent).showVisualModels_.getValue());
                break;
            case 2:
                showBehaviorModels_.setValue((*parent).showBehaviorModels_.getValue());
                break;
            case 3:
                showCollisionModels_.setValue((*parent).showCollisionModels_.getValue());
                break;
            case 4:
                showBoundingCollisionModels_.setValue((*parent).showBoundingCollisionModels_.getValue());
                break;
            case 5:
                showMappings_.setValue((*parent).showMappings_.getValue());
                break;
            case 6:
                showMechanicalMappings_.setValue((*parent).showMechanicalMappings_.getValue());
                break;
            case 7:
                showForceFields_.setValue((*parent).showForceFields_.getValue());
                break;
            case 8:
                showInteractionForceFields_.setValue((*parent).showInteractionForceFields_.getValue());
                break;
            case 9:
                showWireFrame_.setValue((*parent).showWireFrame_.getValue());
                break;
            case 10:
                showNormals_.setValue((*parent).showNormals_.getValue());
                break;
            }
        }
    }
    simulation::Node::updateVisualContext(FILTER);
}

/// Log time spent on an action category, and the concerned object, plus remove the computed time from the parent caller object
void GNode::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
    objectTime[s][parent].tObject -= t;
}

/// Log time spent given a start time, an action category, and the concerned object
GNode::ctime_t GNode::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    const ctime_t t = t1 - t0;
    addTime(t, s, obj, parent);
    return t1;
}

/// Log time spent on an action category and the concerned object
void GNode::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
}

/// Log time spent given a start time, an action category, and the concerned object
GNode::ctime_t GNode::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    const ctime_t t = t1 - t0;
    addTime(t, s, obj);
    return t1;
}

void GNode::addListener(MutationListener* obj)
{
    // make sure we don't add the same listener twice
    Sequence< MutationListener >::iterator it = listener.begin();
    while (it != listener.end() && (*it)!=obj)
        ++it;
    if (it == listener.end())
        listener.add(obj);
}

void GNode::removeListener(MutationListener* obj)
{
    listener.remove(obj);
}

void GNode::notifyAddObject(core::objectmodel::BaseObject* obj)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->addObject(this, obj);
}

void GNode::notifyRemoveObject(core::objectmodel::BaseObject* obj)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->removeObject(this, obj);
}

void GNode::notifyMoveObject(core::objectmodel::BaseObject* obj, GNode* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveObject(prev, this, obj);
}



void create(GNode*& obj, xml::Element<core::objectmodel::BaseNode>* arg)
{
    obj = new GNode();
    obj->parse(arg);
    /*
    // This is no longer necessary as datafields are now used to parse attributes
    obj->setDt(atof(arg->getAttribute("dt","0.01")));
    obj->setTime(atof(arg->getAttribute("time","0.0")));
    obj->setAnimate((atoi(arg->getAttribute("animate","0"))!=0));
    obj->setDebug((atoi(arg->getAttribute("debug","0"))!=0));
    obj->setShowCollisionModels((atoi(arg->getAttribute("showCollisionModels","0"))!=0));
    obj->setShowBoundingCollisionModels((atoi(arg->getAttribute("showBoundingCollisionModels",arg->getAttribute("showCollisionModels","0")))!=0));
    obj->setShowBehaviorModels((atoi(arg->getAttribute("showBehaviorModels","0"))!=0));
    obj->setShowVisualModels((atoi(arg->getAttribute("showVisualModels","1"))!=0));
    obj->setShowMappings((atoi(arg->getAttribute("showMappings","0"))!=0));
    obj->setShowMechanicalMappings((atoi(arg->getAttribute("showMechanicalMappings",arg->getAttribute("showMappings","0")))!=0));
    obj->setShowForceFields((atoi(arg->getAttribute("showForceFields","0"))!=0));
    obj->setShowInteractionForceFields((atoi(arg->getAttribute("showInteractionForceFields",arg->getAttribute("showForceFields","0")))!=0));
    obj->setShowWireFrame((atoi(arg->getAttribute("showWireFrame","0"))!=0));
    obj->setShowNormals((atoi(arg->getAttribute("showNormals","0"))!=0));
    */
}

SOFA_DECL_CLASS(GNode)

helper::Creator<xml::NodeElement::Factory, GNode> GNodeClass("default");

} // namespace tree

} // namespace simulation

} // namespace sofa

