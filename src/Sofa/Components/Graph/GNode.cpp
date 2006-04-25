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
{
}

GNode::GNode()
{
}

GNode::GNode(const std::string& name)
{
    setName(name);
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

/// Add an object. Detect the implemented interfaces and add the object to the corresponding lists.
void GNode::addObject(BaseObject* obj)
{
    obj->setNode(this);
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
}

/// Remove an object
void GNode::removeObject(BaseObject* obj)
{
    if (obj->getNode()==this)
    {
        obj->setNode(NULL);
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
}

/// Connect all objects together. Must be called after each graph modification.
void GNode::init()
{
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

/// Simulation timestep
double GNode::getDt() const { return searchProperties()->dt; }

/// Gravity vector as a pointer to 3 double
const double* GNode::getGravity() const { return searchProperties()->gravity; }

/// Animation flag
bool GNode::getAnimate() const { return searchProperties()->animate; }

/// MultiThreading activated
bool GNode::getMultiThreadSimulation() const { return searchProperties()->multiThreadSimulation; }

/// Display flags: Collision Models
bool GNode::getShowCollisionModels() const { return searchProperties()->showCollisionModels; }

/// Display flags: Behavior Models
bool GNode::getShowBehaviorModels() const { return searchProperties()->showBehaviorModels; }

/// Display flags: Visual Models
bool GNode::getShowVisualModels() const { return searchProperties()->showVisualModels; }

/// Display flags: Mappings
bool GNode::getShowMappings() const { return searchProperties()->showMappings; }

/// Display flags: ForceFields
bool GNode::getShowForceFields() const { return searchProperties()->showForceFields; }

/// Simulation timestep
void GNode::setDt(double val)
{
    newProperties()->dt = val;
}

/// Gravity vector as a pointer to 3 double
void GNode::setGravity(const double* val)
{
    newProperties()->gravity[0] = val[0];
    newProperties()->gravity[1] = val[1];
    newProperties()->gravity[2] = val[2];
}

/// Animation flag
void GNode::setAnimate(bool val)
{
    newProperties()->animate = val;
}

/// MultiThreading activated
void GNode::setMultiThreadSimulation(bool val)
{
    newProperties()->multiThreadSimulation = val;
}

/// Display flags: Collision Models
void GNode::setShowCollisionModels(bool val)
{
    newProperties()->showCollisionModels = val;
}

/// Display flags: Behavior Models
void GNode::setShowBehaviorModels(bool val)
{
    newProperties()->showBehaviorModels = val;
}

/// Display flags: Visual Models
void GNode::setShowVisualModels(bool val)
{
    newProperties()->showVisualModels = val;
}

/// Display flags: Mappings
void GNode::setShowMappings(bool val)
{
    newProperties()->showMappings = val;
}

/// Display flags: ForceFields
void GNode::setShowForceFields(bool val)
{
    newProperties()->showForceFields = val;
}

GNode::Properties::Properties()
{
    dt = 0.04;
    gravity[0] = 0;
    gravity[1] = -9.8;
    gravity[2] = 0;
    animate = false;
    showCollisionModels = true;
    showBehaviorModels = true;
    showVisualModels = true;
    showMappings = true;
    showForceFields = true;
    multiThreadSimulation = false;
}

GNode::Properties GNode::Properties::defaultProperties;

const GNode::Properties* GNode::searchProperties() const
{
    if (!properties.empty()) return properties;
    else if (!parent.empty()) return parent->searchProperties();
    else return &Properties::defaultProperties;
}

GNode::Properties* GNode::newProperties()
{
    if (properties.empty()) properties.add(new Properties(*searchProperties()));
    return properties;
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

void create(GNode*& obj, XML::Node<Abstract::BaseNode>* /*arg*/)
{
    obj = new GNode();
}

SOFA_DECL_CLASS(GNode)

Common::Creator<XML::NodeNode::Factory, GNode> GNodeClass("default");

} // namespace Graph

} // namespace Components

} // namespace Sofa
