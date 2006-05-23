#include "GNode.h"
#include "Action.h"
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

GNode::~GNode()
{}


GNode::GNode(const std::string& name, GNode* parent)
    : debug_(false)
{
    setName(name);
    if( parent )
        parent->addChild(this);
}

/// Add a child node
BaseNode* GNode::addChild(GNode* node)
{
    child.add(node);
    node->parent.add(this);
    return this;
}

/// Remove a child
void GNode::removeChild(GNode* node)
{
    child.remove(node);
    node->parent.remove(this);
}

/// Add a child node
BaseNode* GNode::addChild(BaseNode* node)
{
    this->addChild(dynamic_cast<GNode*>(node));
    return this;
}

/// Remove a child node
void GNode::removeChild(BaseNode* node)
{
    this->removeChild(dynamic_cast<GNode*>(node));
}

Core::Context* GNode::getContext()
{
    return &context_;
}
const Core::Context* GNode::getContext() const
{
    return &context_;
}

Sofa::Core::Context* GNode::getParentContext()
{
    if( GNode* p = dynamic_cast<GNode*>(&(*parent)) )
    {
        return p->getContext();
    }
    else return NULL;
}

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
GNode* GNode::addObject(BaseObject* obj)
{
    obj->setContext(getContext());
    object.add(obj);
    mechanicalModel.add(dynamic_cast< BasicMechanicalModel* >(obj));
    if (!mechanicalMapping.add(dynamic_cast< BasicMechanicalMapping* >(obj)))
        mapping.add(dynamic_cast< BasicMapping* >(obj));
    solver.add(dynamic_cast< OdeSolver* >(obj));
    mass.add(dynamic_cast< Mass* >(obj));
    topology.add(dynamic_cast< Topology* >(obj));
    if (!interactionForceField.add(dynamic_cast< InteractionForceField* >(obj)))
        forceField.add(dynamic_cast< ForceField* >(obj));
    constraint.add(dynamic_cast< Constraint* >(obj));
    behaviorModel.add(dynamic_cast< BehaviorModel* >(obj));
    visualModel.add(dynamic_cast< VisualModel* >(obj));
    collisionModel.add(dynamic_cast< CollisionModel* >(obj));
    contextObject.add(dynamic_cast<ContextObject* >(obj));

    return this;
}

/// Remove an object
void GNode::removeObject(BaseObject* obj)
{
    if (obj->getContext()==getContext())
    {
        obj->setContext(NULL);
    }
    object.remove(obj);
    mechanicalModel.remove(dynamic_cast< BasicMechanicalModel* >(obj));
    mechanicalMapping.remove(dynamic_cast< BasicMechanicalMapping* >(obj));
    solver.remove(dynamic_cast< OdeSolver* >(obj));
    mass.remove(dynamic_cast< Mass* >(obj));
    topology.remove(dynamic_cast< Topology* >(obj));
    forceField.remove(dynamic_cast< ForceField* >(obj));
    interactionForceField.remove(dynamic_cast< InteractionForceField* >(obj));
    constraint.remove(dynamic_cast< Constraint* >(obj));
    mapping.remove(dynamic_cast< BasicMapping* >(obj));
    behaviorModel.remove(dynamic_cast< BehaviorModel* >(obj));
    visualModel.remove(dynamic_cast< VisualModel* >(obj));
    collisionModel.remove(dynamic_cast< CollisionModel* >(obj));
    contextObject.remove(dynamic_cast<ContextObject* >(obj));
}



/// Connect all objects together. Must be called after each graph modification.
void GNode::init()
{
    updateContext();
    //cerr<<"GNode::init()"<<endl;
    if( !mechanicalMapping.empty() )
    {
        mechanicalMapping->propagateX();
        mechanicalMapping->propagateV();
    }

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
    if( getParentContext() != NULL )
    {
        context_ = *getParentContext();
        //cerr<<"node "<<getName()<<", copy context"<<endl;
    }
    else
    {
        context_ = Context::getDefault() ;
        //cerr<<"node "<<getName()<<", apply default context"<<endl;
    }

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
        cerr<<"GNode::updateContext, node = "<<getName()<<", updated context = "<< *this->getContext() << endl;
    }
}


/// Execute a recursive action starting from this node
void GNode::execute(Action* action)
{
    if(action->processNodeTopDown(this) != Action::RESULT_PRUNE)
    {
        for(ChildIterator it = child.begin(); it != child.end(); ++it)
        {
            (*it)->execute(action);
        }
    }
    action->processNodeBottomUp(this);
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

void create(GNode*& obj, XML::Node<Abstract::BaseNode>* /*arg*/)
{
    obj = new GNode();
}

SOFA_DECL_CLASS(GNode)

Common::Creator<XML::NodeNode::Factory, GNode> GNodeClass("default");

} // namespace Graph

} // namespace Components

} // namespace Sofa


