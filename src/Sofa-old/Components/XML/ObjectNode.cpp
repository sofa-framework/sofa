#include "ObjectNode.h"
#include "Node.inl"

namespace Sofa
{

namespace Components
{

namespace XML
{

using namespace Common;

//template class Factory< std::string, Abstract::BaseObject, Node<Abstract::BaseObject*>* >;

ObjectNode::ObjectNode(const std::string& name, const std::string& type, BaseNode* parent)
    : Node<Abstract::BaseObject>(name, type, parent)
{
}

ObjectNode::~ObjectNode()
{
}

bool ObjectNode::initNode()
{
    if (!Node<Abstract::BaseObject>::initNode()) return false;
    if (getObject()!=NULL)
    {
        Abstract::BaseContext* ctx = dynamic_cast<Abstract::BaseContext*>(getParent()->getBaseObject());
        if (ctx!=NULL)
        {
            std::cout << "Adding Object "<<getName()<<" to "<<ctx->getName()<<std::endl;
            ctx->addObject(getObject());
        }
    }
    return true;
}

SOFA_DECL_CLASS(Object)

Creator<BaseNode::NodeFactory, ObjectNode> ObjectNodeClass("Object");
Creator<BaseNode::NodeFactory, ObjectNode> PropertyNodeClass("Property");
Creator<BaseNode::NodeFactory, ObjectNode> MechanicalNodeClass("Mechanical");
Creator<BaseNode::NodeFactory, ObjectNode> TopologyNodeClass("Topology");
Creator<BaseNode::NodeFactory, ObjectNode> ForceFieldNodeClass("ForceField");
Creator<BaseNode::NodeFactory, ObjectNode> InteractionForceFieldNodeClass("InteractionForceField");
Creator<BaseNode::NodeFactory, ObjectNode> MassNodeClass("Mass");
Creator<BaseNode::NodeFactory, ObjectNode> ConstraintNodeClass("Constraint");
Creator<BaseNode::NodeFactory, ObjectNode> MappingNodeClass("Mapping");
Creator<BaseNode::NodeFactory, ObjectNode> SolverNodeClass("Solver");
Creator<BaseNode::NodeFactory, ObjectNode> CollisionNodeClass("Collision");
Creator<BaseNode::NodeFactory, ObjectNode> VisualNodeClass("Visual");
Creator<BaseNode::NodeFactory, ObjectNode> BehaviorNodeClass("Behavior");

const char* ObjectNode::getClass() const
{
    return ObjectNodeClass.c_str();
}

} // namespace XML

} // namespace Components

} // namespace Sofa
