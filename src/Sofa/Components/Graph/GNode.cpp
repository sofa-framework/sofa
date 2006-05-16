#include "GNode.h"
#include "Action.h"
#include "Sofa/Components/XML/NodeNode.h"

namespace Sofa
{

namespace Components
{

namespace Graph
{

GNode::~GNode()
{}

GNode::GNode()
{}

GNode::GNode(const std::string& name)
{
    setName(name);
}

GNode::GNode(const std::string& name, GNode* parent)
{
    setName(name);
    parent->addChild(this);
}

/// Add a child node
void GNode::addChild(GNode* node)
{
    child.add(node);
    node->parent.add(this);
}

/// Remove a child
void GNode::removeChild(GNode* node)
{
    child.remove(node);
    node->parent.remove(this);
}

Sofa::Core::Context* GNode::getParentContext()
{
    return dynamic_cast<GNode*>(&(*parent));
    // uglyssimo! and could be more efficient
}

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
void GNode::addObject(BaseObject* obj)
{
    obj->setContext(this);
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
}

/// Remove an object
void GNode::removeObject(BaseObject* obj)
{
    if (obj->getContext()==this)
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
//     for( Sequence<BaseObject>::iterator it = object.begin(); it!=object.end(); it++ ){
// 	(*it)->setContext(this);
//     }

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



// GNode::Properties::Properties()
// {
// 	dt = 0.04;
// 	gravity[0] = 0;
// 	gravity[1] = -9.8;
// 	gravity[2] = 0;
// 	animate = false;
// 	showCollisionModels = true;
// 	showBehaviorModels = true;
// 	showVisualModels = true;
// 	showMappings = true;
// 	showForceFields = true;
// 	multiThreadSimulation = false;
// }

// GNode::Properties GNode::Properties::defaultProperties;
//
// const GNode::Properties* GNode::searchProperties() const
// {
// 	if (!properties.empty()) return properties;
// 	else if (!parent.empty()) return parent->searchProperties();
// 	else return &Properties::defaultProperties;
// }
//
// GNode::Properties* GNode::newProperties()
// {
// 	if (properties.empty()) properties.add(new Properties(*searchProperties()));
// 	return properties;
// }

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

void create(GNode*& obj, XML::Node<Abstract::BaseNode>* /*arg*/)
{
    obj = new GNode();
}

SOFA_DECL_CLASS(GNode)

Common::Creator<XML::NodeNode::Factory, GNode> GNodeClass("default");

} // namespace Graph

} // namespace Components

} // namespace Sofa


