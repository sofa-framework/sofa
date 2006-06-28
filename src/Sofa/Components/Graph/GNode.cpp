#include "GNode.h"
#include "Action.h"
#include "MutationListener.h"
#include "Sofa/Components/XML/NodeNode.h"
#include <iostream>
using std::cerr;
using std::endl;

namespace Sofa
{

namespace Components
{

namespace Graph
{

GNode::GNode(const std::string& name, GNode* parent)
    : debug_(false)
{
    setName(name);
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
void GNode::addChild(BaseNode* node)
{
    this->addChild(dynamic_cast<GNode*>(node));
}

/// Remove a child node
void GNode::removeChild(BaseNode* node)
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

BaseContext* GNode::getContext()
{
    return this;
}
const BaseContext* GNode::getContext() const
{
    return this;
}
/*
Sofa::Core::Context* GNode::getParentContext()
{
	if( GNode* p = dynamic_cast<GNode*>(&(*parent)) ){
		return p->getContext();
	}
	else return NULL;
}
*/

/// Mechanical Degrees-of-Freedom
Abstract::BaseObject* GNode::getMechanicalModel() const
{
    return this->mechanicalModel;
}

/// Topology
Abstract::BaseObject* GNode::getTopology() const
{
    return this->topology;
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
    mechanicalModel.add(dynamic_cast< BasicMechanicalModel* >(obj));
    if (!mechanicalMapping.add(dynamic_cast< BasicMechanicalMapping* >(obj)))
        mapping.add(dynamic_cast< BasicMapping* >(obj));
    solver.add(dynamic_cast< OdeSolver* >(obj));
    mass.add(dynamic_cast< BasicMass* >(obj));
    topology.add(dynamic_cast< Topology* >(obj));
    if (!interactionForceField.add(dynamic_cast< InteractionForceField* >(obj)))
        forceField.add(dynamic_cast< BasicForceField* >(obj));
    constraint.add(dynamic_cast< BasicConstraint* >(obj));
    behaviorModel.add(dynamic_cast< BehaviorModel* >(obj));
    visualModel.add(dynamic_cast< VisualModel* >(obj));
    collisionModel.add(dynamic_cast< CollisionModel* >(obj));
    contextObject.add(dynamic_cast< ContextObject* >(obj));
    collisionPipeline.add(dynamic_cast< Collision::Pipeline* >(obj));
}

/// Remove an object
void GNode::doRemoveObject(BaseObject* obj)
{
    if (obj->getContext()==this)
    {
        obj->setContext(NULL);
    }
    object.remove(obj);
    mechanicalModel.remove(dynamic_cast< BasicMechanicalModel* >(obj));
    mechanicalMapping.remove(dynamic_cast< BasicMechanicalMapping* >(obj));
    solver.remove(dynamic_cast< OdeSolver* >(obj));
    mass.remove(dynamic_cast< BasicMass* >(obj));
    topology.remove(dynamic_cast< Topology* >(obj));
    forceField.remove(dynamic_cast< BasicForceField* >(obj));
    interactionForceField.remove(dynamic_cast< InteractionForceField* >(obj));
    constraint.remove(dynamic_cast< BasicConstraint* >(obj));
    mapping.remove(dynamic_cast< BasicMapping* >(obj));
    behaviorModel.remove(dynamic_cast< BehaviorModel* >(obj));
    visualModel.remove(dynamic_cast< VisualModel* >(obj));
    collisionModel.remove(dynamic_cast< CollisionModel* >(obj));
    contextObject.remove(dynamic_cast<ContextObject* >(obj));
    collisionPipeline.remove(dynamic_cast< Collision::Pipeline* >(obj));
}



/// Connect all objects together. Must be called after each graph modification.
void GNode::init()
{
    //cerr<<"GNode::init()"<<endl;

    for (Sequence<BaseObject>::iterator it = object.begin(); it != object.end(); it++)
    {
        (*it)->init();
    }

    updateContext();
    /*
    if( !mechanicalMapping.empty() ){
    mechanicalMapping->propagateX();
    mechanicalMapping->propagateV();
    }
    */
    for (Sequence<GNode>::iterator it = child.begin(); it != child.end(); it++)
    {
        (*it)->init();
    }
}

/// Get parent node (or NULL if no hierarchy or for root node)
BaseNode* GNode::getParent()
{
    return parent;
}

/// Get parent node (or NULL if no hierarchy or for root node)
const BaseNode* GNode::getParent() const
{
    return parent;
}

void GNode::updateContext()
{
    if( getParent() != NULL )
    {
        copyContext(*parent);
        //cerr<<"node "<<getName()<<", copy context"<<endl;
    }
    //else {
    //	*static_cast<Core::Context*>(this) = Core::Context() ;
    //	//cerr<<"node "<<getName()<<", apply default context"<<endl;
    //}

    /*	if( debug_ ){
    	   cerr<<"GNode::updateContext, node = "<<getName()<<", incoming context = "<< *this->getContext() << endl;
    	}*/
    // Apply local modifications to the context
    for( unsigned i=0; i<contextObject.size(); ++i )
    {
        contextObject[i]->apply();
        /*		if( debug_ ){
        		   cerr<<"GNode::updateContext, modified by node = "<<contextObject[i]->getName()<<", new context = "<< *this->getContext() << endl;
        		}*/
    }
// 	if( !mechanicalModel.empty() ){
// 	   mechanicalModel->updateContext(&context_);
// 	}
    if( debug_ )
    {
        cerr<<"GNode::updateContext, node = "<<getName()<<", updated context = "<< *static_cast<Core::Context*>(this) << endl;
    }
}


/// Execute a recursive action starting from this node
void GNode::executeAction(Action* action)
{
    if (getLogTime())
    {
        const ctime_t t0 = Thread::CTime::getTime();
        ctime_t tChild = 0;
        if(action->processNodeTopDown(this) != Action::RESULT_PRUNE)
        {
            ctime_t ct0 = Thread::CTime::getTime();
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                (*it)->executeAction(action);
            }
            tChild = Thread::CTime::getTime() - ct0;
        }
        action->processNodeBottomUp(this);
        ctime_t tTree = Thread::CTime::getTime() - t0;
        ctime_t tNode = tTree - tChild;
        totalTime.tNode += tNode;
        totalTime.tTree += tTree;
        ++totalTime.nVisit;
        Timer& t = actionTime[action->getCategoryName()];
        t.tNode += tNode;
        t.tTree += tTree;
        ++t.nVisit;
    }
    else
    {
        if(action->processNodeTopDown(this) != Action::RESULT_PRUNE)
        {
            for(ChildIterator it = child.begin(); it != child.end(); ++it)
            {
                (*it)->executeAction(action);
            }
        }
        action->processNodeBottomUp(this);
    }
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

bool GNode::getLogTime() const
{
    return logTime_;
}

GNode::ctime_t GNode::getTimeFreq() const
{
    return Thread::CTime::getTicksPerSec();
}

void GNode::resetTime()
{
    totalTime.nVisit = 0;
    totalTime.tNode = 0;
    totalTime.tTree = 0;
    actionTime.clear();
}

void GNode::addListener(MutationListener* obj)
{
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

void GNode::notifyAddObject(Abstract::BaseObject* obj)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->addObject(this, obj);
}

void GNode::notifyRemoveObject(Abstract::BaseObject* obj)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->removeObject(this, obj);
}

void GNode::notifyMoveChild(GNode* node, GNode* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveChild(prev, this, node);
}

void GNode::notifyMoveObject(Abstract::BaseObject* obj, GNode* prev)
{
    for (Sequence<MutationListener>::iterator it = listener.begin(); it != listener.end(); ++it)
        (*it)->moveObject(prev, this, obj);
}

void create(GNode*& obj, XML::Node<Abstract::BaseNode>* arg)
{
    obj = new GNode();
    obj->setDt(atof(arg->getAttribute("dt","0.01")));
    obj->setAnimate((atoi(arg->getAttribute("animate","0"))!=0));
    obj->setDebug((atoi(arg->getAttribute("debug","0"))!=0));
    obj->setShowCollisionModels((atoi(arg->getAttribute("showCollisionModels","0"))!=0));
    obj->setShowBehaviorModels((atoi(arg->getAttribute("showBehaviorModels","0"))!=0));
    obj->setShowVisualModels((atoi(arg->getAttribute("showVisualModels","1"))!=0));
    obj->setShowMappings((atoi(arg->getAttribute("showMappings","0"))!=0));
    obj->setShowForceFields((atoi(arg->getAttribute("showForceFields","0"))!=0));
    obj->setShowWireFrame((atoi(arg->getAttribute("showWireFrame","0"))!=0));
    obj->setShowNormals((atoi(arg->getAttribute("showNormals","0"))!=0));

}

SOFA_DECL_CLASS(GNode)

Common::Creator<XML::NodeNode::Factory, GNode> GNodeClass("default");

} // namespace Graph

} // namespace Components

} // namespace Sofa


