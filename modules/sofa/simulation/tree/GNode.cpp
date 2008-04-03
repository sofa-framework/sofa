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
#include <sofa/simulation/tree/Visitor.h>
#include <sofa/simulation/tree/PropagateEventVisitor.h>
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
    : debug_(false), logTime_(false)
{
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    setName(name);
    if( parent )
        parent->addChild(this);
}

GNode::~GNode()
{}

/// Update the parameters of the GNode
void GNode::reinit()
{
    GNode *context = static_cast<GNode *>(getContext());
    for (GNode::ChildIterator it= context->child.begin(); it != context->child.end(); ++it)
    {
        (*it)->is_activated.setValue(is_activated.getValue());
        (*it)->reinit();
    }
}

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

/// Move an object from another node
void GNode::moveObject(BaseObject* obj)
{
    GNode* prev = dynamic_cast<GNode*>(obj->getContext());
    if (prev==NULL)
    {
        obj->getContext()->removeObject(obj);
        addObject(obj);
    }
    else
    {
        notifyMoveObject(obj,prev);
        prev->doRemoveObject(obj);
        doAddObject(obj);
    }
}

core::objectmodel::BaseContext* GNode::getContext()
{
    return this;
}
const core::objectmodel::BaseContext* GNode::getContext() const
{
    return this;
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
        return getObject(newpath);
    }
    else if (std::string(path,0,3)==std::string("../"))
    {
        std::string newpath = std::string(path, 3);
        while (!newpath.empty() && path[0] == '/')
            newpath.erase(0);
        if (parent) return parent->getObject(newpath);
        else return getObject(newpath);
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
            core::objectmodel::BaseObject* obj = getObject(name);
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
            parent->getObjects(class_info, container, dir);
            return;
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
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
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

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
bool GNode::addObject(BaseObject* obj)
{
    notifyAddObject(obj);
    doAddObject(obj);
    return true;
}

/// Remove an object
bool GNode::removeObject(BaseObject* obj)
{
    notifyRemoveObject(obj);
    doRemoveObject(obj);
    return true;
}

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
void GNode::doAddObject(BaseObject* obj)
{
    notifyAddObject(obj);
    obj->setContext(this);
    object.add(obj);
    masterSolver.add(dynamic_cast< core::componentmodel::behavior::MasterSolver* >(obj));
    solver.add(dynamic_cast< core::componentmodel::behavior::OdeSolver* >(obj));
    mechanicalState.add(dynamic_cast< core::componentmodel::behavior::BaseMechanicalState* >(obj));
    if (!mechanicalMapping.add(dynamic_cast< core::componentmodel::behavior::BaseMechanicalMapping* >(obj)))
        mapping.add(dynamic_cast< core::BaseMapping* >(obj));
    mass.add(dynamic_cast< core::componentmodel::behavior::BaseMass* >(obj));
    topology.add(dynamic_cast< core::componentmodel::topology::Topology* >(obj));
    basicTopology.add(dynamic_cast< core::componentmodel::topology::BaseTopology* >(obj));
    meshTopology.add(dynamic_cast< core::componentmodel::topology::BaseMeshTopology* >(obj));
    shader.add(dynamic_cast< sofa::core::Shader* >(obj));

    if (!interactionForceField.add(dynamic_cast< core::componentmodel::behavior::InteractionForceField* >(obj)))
        forceField.add(dynamic_cast< core::componentmodel::behavior::BaseForceField* >(obj));
    constraint.add(dynamic_cast< core::componentmodel::behavior::BaseConstraint* >(obj));
    behaviorModel.add(dynamic_cast< core::BehaviorModel* >(obj));
    visualModel.add(dynamic_cast< core::VisualModel* >(obj));
    collisionModel.add(dynamic_cast< core::CollisionModel* >(obj));
    contextObject.add(dynamic_cast< core::objectmodel::ContextObject* >(obj));
    collisionPipeline.add(dynamic_cast< core::componentmodel::collision::Pipeline* >(obj));
    actionScheduler.add(dynamic_cast< VisitorScheduler* >(obj));
}

/// Remove an object
void GNode::doRemoveObject(BaseObject* obj)
{
    if (obj->getContext()==this)
    {
        obj->setContext(NULL);
    }
    object.remove(obj);
    masterSolver.remove(dynamic_cast< core::componentmodel::behavior::MasterSolver* >(obj));
    solver.remove(dynamic_cast< core::componentmodel::behavior::OdeSolver* >(obj));
    mechanicalState.remove(dynamic_cast< core::componentmodel::behavior::BaseMechanicalState* >(obj));
    mechanicalMapping.remove(dynamic_cast< core::componentmodel::behavior::BaseMechanicalMapping* >(obj));
    mass.remove(dynamic_cast< core::componentmodel::behavior::BaseMass* >(obj));
    topology.remove(dynamic_cast< core::componentmodel::topology::Topology* >(obj));
    basicTopology.remove(dynamic_cast< core::componentmodel::topology::BaseTopology* >(obj));
    meshTopology.remove(dynamic_cast< core::componentmodel::topology::BaseMeshTopology* >(obj));
    shader.remove(dynamic_cast<sofa::core::Shader* >(obj));

    forceField.remove(dynamic_cast< core::componentmodel::behavior::BaseForceField* >(obj));
    interactionForceField.remove(dynamic_cast< core::componentmodel::behavior::InteractionForceField* >(obj));
    constraint.remove(dynamic_cast< core::componentmodel::behavior::BaseConstraint* >(obj));
    mapping.remove(dynamic_cast< core::BaseMapping* >(obj));
    behaviorModel.remove(dynamic_cast< core::BehaviorModel* >(obj));
    visualModel.remove(dynamic_cast< core::VisualModel* >(obj));
    collisionModel.remove(dynamic_cast< core::CollisionModel* >(obj));
    contextObject.remove(dynamic_cast<core::objectmodel::ContextObject* >(obj));
    collisionPipeline.remove(dynamic_cast< core::componentmodel::collision::Pipeline* >(obj));
    actionScheduler.remove(dynamic_cast< VisitorScheduler* >(obj));
    // Remove references to this object in time log tables
    if (!objectTime.empty())
    {
        for (std::map<std::string, std::map<core::objectmodel::BaseObject*, simulation::tree::GNode::ObjectTimer> >::iterator it = objectTime.begin(); it != objectTime.end(); ++it)
        {
            it->second.erase(obj);
        }
    }
}

void GNode::initVisualContext()
{
    if (getParent() != NULL)
    {
        if (showVisualModels_.getValue() == -1)            showVisualModels_.setValue(static_cast<GNode *>(getParent())->showVisualModels_.getValue());
        if (showBehaviorModels_.getValue() == -1)          showBehaviorModels_.setValue(static_cast<GNode *>(getParent())->showBehaviorModels_.getValue());
        if (showCollisionModels_.getValue() == -1)         showCollisionModels_.setValue(static_cast<GNode *>(getParent())->showCollisionModels_.getValue());
        if (showBoundingCollisionModels_.getValue() == -1) showBoundingCollisionModels_.setValue(static_cast<GNode *>(getParent())->showBoundingCollisionModels_.getValue());
        if (showMappings_.getValue() == -1)                showMappings_.setValue(static_cast<GNode *>(getParent())->showMappings_.getValue());
        if (showMechanicalMappings_.getValue() == -1)      showMechanicalMappings_.setValue(static_cast<GNode *>(getParent())->showMechanicalMappings_.getValue());
        if (showForceFields_.getValue() == -1)             showForceFields_.setValue(static_cast<GNode *>(getParent())->showForceFields_.getValue());
        if (showInteractionForceFields_.getValue() == -1)  showInteractionForceFields_.setValue(static_cast<GNode *>(getParent())->showInteractionForceFields_.getValue());
        if (showWireFrame_.getValue() == -1)               showWireFrame_.setValue(static_cast<GNode *>(getParent())->showWireFrame_.getValue());
        if (showNormals_.getValue() == -1)                 showNormals_.setValue(static_cast<GNode *>(getParent())->showNormals_.getValue());
    }
}

void GNode::setDefaultVisualContextValue()
{
    if (showVisualModels_.getValue() == -1)            showVisualModels_.setValue(true);
    if (showBehaviorModels_.getValue() == -1)          showBehaviorModels_.setValue(false);
    if (showCollisionModels_.getValue() == -1)         showCollisionModels_.setValue(false);
    if (showBoundingCollisionModels_.getValue() == -1) showBoundingCollisionModels_.setValue(false);
    if (showMappings_.getValue() == -1)                showMappings_.setValue(false);
    if (showMechanicalMappings_.getValue() == -1)      showMechanicalMappings_.setValue(false);
    if (showForceFields_.getValue() == -1)             showForceFields_.setValue(false);
    if (showInteractionForceFields_.getValue() == -1)  showInteractionForceFields_.setValue(false);
    if (showWireFrame_.getValue() == -1)               showWireFrame_.setValue(false);
    if (showNormals_.getValue() == -1)                 showNormals_.setValue(false);
}

void GNode::initialize()
{
    //cerr<<"GNode::initialize()"<<endl;

    initVisualContext();
    // Put the OdeSolver, if any, in first position. This makes sure that the OdeSolver component is initialized only when all its sibling and children components are already initialized.
    /// @todo Putting the solver first means that it will be initialized *before* any sibling or childrens. Is that what we want? -- Jeremie A.
    Sequence<BaseObject>::iterator i=object.begin(), iend=object.end();
    for( ; i!=iend && dynamic_cast<core::componentmodel::behavior::OdeSolver*>(*i)==NULL; i++ ) // find the OdeSolver
    {}
    if( i!=iend && !object.empty() ) // found
    {
        // put it first
        // BUGFIX 01/12/06 (Jeremie A.): do not modify the order of the other objects
        // object.swap( i, object.begin() );
        while (i!=object.begin())
        {
            Sequence<BaseObject>::iterator i2 = i;
            --i;
            object.swap(i, i2);
        }
    }

    //
    updateSimulationContext();

    // this is now done by the InitVisitor
    //for (Sequence<GNode>::iterator it = child.begin(); it != child.end(); it++) {
    //    (*it)->init();
    //}
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

void GNode::updateContext()
{
    if( getParent() != NULL )
    {
        copyContext(*parent);
        //cerr<<"node "<<getName()<<", copy context, time = "<<getTime()<<endl;
    }
    //else
    //	cerr<<"node "<<getName()<<", time = "<<getTime()<<endl;
    //else {
    //	*static_cast<Core::Context*>(this) = Core::Context() ;
    //	//cerr<<"node "<<getName()<<", apply default context"<<endl;
    //}


    //if( debug_ ) cerr<<"GNode::updateContext, node = "<<getName()<<", incoming context = "<< *this->getContext() << endl;

    // Apply local modifications to the context
    if (getLogTime())
    {
        for( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
            //cerr<<"GNode::updateContext, modified by node = "<<contextObject[i]->getName()<< endl;
        }
    }
    else
    {
        for( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
            //cerr<<"GNode::updateContext, modified by node = "<<contextObject[i]->getName()<<endl;
        }
    }
//	if( !mechanicalModel.empty() ) {
//		mechanicalModel->updateContext(&context_);
//	}


    // project the gravity to the local coordinate system
    /*        getContext()->setGravity( getContext()->getLocalFrame().backProjectVector(getContext()->getWorldGravity()) );*/

    if( debug_ ) std::cerr<<"GNode::updateContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

void GNode::updateSimulationContext()
{
    if( getParent() != NULL )
    {
        copySimulationContext(*parent);
    }

    // Apply local modifications to the context
    if (getLogTime())
    {
        for( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    else
    {
        for( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    if( debug_ ) std::cerr<<"GNode::updateSimulationContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}

void GNode::updateVisualContext(int FILTER)
{
    if( getParent() != NULL )
    {
        if (!FILTER)
            copyVisualContext(*parent);
        else
        {
            switch(FILTER)
            {
            case 1:  showVisualModels_.setValue((*parent).showVisualModels_.getValue());  break;
            case 2:  showBehaviorModels_.setValue((*parent).showBehaviorModels_.getValue());  break;
            case 3:  showCollisionModels_.setValue((*parent).showCollisionModels_.getValue());  break;
            case 4:  showBoundingCollisionModels_.setValue((*parent).showBoundingCollisionModels_.getValue());  break;
            case 5:  showMappings_.setValue((*parent).showMappings_.getValue());  break;
            case 6:  showMechanicalMappings_.setValue((*parent).showMechanicalMappings_.getValue());  break;
            case 7:  showForceFields_.setValue((*parent).showForceFields_.getValue());  break;
            case 8:  showInteractionForceFields_.setValue((*parent).showInteractionForceFields_.getValue()); break;
            case 9:  showWireFrame_.setValue((*parent).showWireFrame_.getValue()); break;
            case 10: showNormals_.setValue((*parent).showNormals_.getValue()); break;
            }
        }
    }
    // Apply local modifications to the context
    if (getLogTime())
    {
        for( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    else
    {
        for( unsigned i=0; i<contextObject.size(); ++i )
        {
            contextObject[i]->init();
            contextObject[i]->apply();
        }
    }
    if( debug_ ) std::cerr<<"GNode::updateVisualContext, node = "<<getName()<<", updated context = "<< *static_cast<core::objectmodel::Context*>(this) << endl;
}


/// Execute a recursive action starting from this node
void GNode::executeVisitor(Visitor* action)
{
    if (!this->is_activated.getValue()) return;
    if (actionScheduler)
        actionScheduler->executeVisitor(this,action);
    else
        doExecuteVisitor(action);
}

/// Execute a recursive action starting from this node
/// This method bypass the actionScheduler of this node if any.
void GNode::doExecuteVisitor(Visitor* action)
{
    if (getLogTime())
    {
        const ctime_t t0 = CTime::getTime();
        ctime_t tChild = 0;
        actionStack.push(action);
        if(action->processNodeTopDown(this) != Visitor::RESULT_PRUNE)
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
            Visitor* prev = actionStack.top();
            NodeTimer& t = actionTime[prev->getCategoryName()];
            t.tNode -= tTree;
            t.tTree -= tTree;
        }
    }
    else
    {
        if(action->processNodeTopDown(this) != Visitor::RESULT_PRUNE)
        {
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                (*it)->executeVisitor(action);
            }
        }
        action->processNodeBottomUp(this);
    }
}

/// Find an object given its name
core::objectmodel::BaseObject* GNode::getObject(const std::string& name) const
{
    for (ObjectIterator it = object.begin(), itend = object.end(); it != itend; ++it)
        if ((*it)->getName() == name)
            return *it;
    return NULL;
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

/// Propagate an event
void GNode::propagateEvent( Event* event )
{
    PropagateEventVisitor act(event);
    this->executeVisitor(&act);
}

GNode* GNode::setDebug(bool b)
{
    debug_=b;
    return this;
}

bool GNode::getDebug() const
{
    return debug_;
}

void GNode::setLogTime(bool b)
{
    logTime_=b;
}

GNode::ctime_t GNode::getTimeFreq() const
{
    return CTime::getTicksPerSec();
}

void GNode::resetTime()
{
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    actionTime.clear();
    objectTime.clear();
}

/// Log time spent on an action category, and the concerned object, plus remove the computed time from the parent caller object
void GNode::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
    objectTime[s][parent].tObject -= t;
}

/// Log time spent on an action category and the concerned object
void GNode::addTime(ctime_t t, const std::string& s, core::objectmodel::BaseObject* obj)
{
    ObjectTimer& timer = objectTime[s][obj];
    timer.tObject += t;
    ++ timer.nVisit;
}

/// Measure start time
GNode::ctime_t GNode::startTime() const
{
    if (!getLogTime()) return 0;
    return CTime::getTime();
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

/// Log time spent given a start time, an action category, and the concerned object
GNode::ctime_t GNode::endTime(ctime_t t0, const std::string& s, core::objectmodel::BaseObject* obj, core::objectmodel::BaseObject* parent)
{
    if (!getLogTime()) return 0;
    const ctime_t t1 = CTime::getTime();
    const ctime_t t = t1 - t0;
    addTime(t, s, obj, parent);
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

void GNode::notifyMoveChild(GNode* node, GNode* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveChild(prev, this, node);
}

void GNode::notifyMoveObject(core::objectmodel::BaseObject* obj, GNode* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveObject(prev, this, obj);
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

