#include <sofa/simulation/tree/xml/ObjectElement.h>
#include <sofa/simulation/tree/xml/Element.inl>

namespace sofa
{

namespace simulation
{

namespace tree
{

namespace xml
{

using namespace sofa::defaulttype;
using helper::Creator;

//template class Factory< std::string, objectmodel::BaseObject, Node<objectmodel::BaseObject*>* >;

ObjectElement::ObjectElement(const std::string& name, const std::string& type, BaseElement* parent)
    : Element<core::objectmodel::BaseObject>(name, type, parent)
{
}

ObjectElement::~ObjectElement()
{
}

bool ObjectElement::initNode()
{
    if (!Element<core::objectmodel::BaseObject>::initNode()) return false;
    if (getObject()!=NULL)
    {
        core::objectmodel::BaseContext* ctx = dynamic_cast<core::objectmodel::BaseContext*>(getParent()->getBaseObject());
        if (ctx!=NULL)
        {
            std::cout << "Adding Object "<<getName()<<" to "<<ctx->getName()<<std::endl;
            ctx->addObject(getObject());
        }
    }
    return true;
}

SOFA_DECL_CLASS(Object)

Creator<BaseElement::NodeFactory, ObjectElement> ObjectNodeClass("Object");
Creator<BaseElement::NodeFactory, ObjectElement> PropertyNodeClass("Property");
Creator<BaseElement::NodeFactory, ObjectElement> MechanicalNodeClass("Mechanical");
Creator<BaseElement::NodeFactory, ObjectElement> TopologyNodeClass("Topology");
Creator<BaseElement::NodeFactory, ObjectElement> ForceFieldNodeClass("ForceField");
Creator<BaseElement::NodeFactory, ObjectElement> InteractionForceFieldNodeClass("InteractionForceField");
Creator<BaseElement::NodeFactory, ObjectElement> MassNodeClass("Mass");
Creator<BaseElement::NodeFactory, ObjectElement> ConstraintNodeClass("Constraint");
Creator<BaseElement::NodeFactory, ObjectElement> MappingNodeClass("Mapping");
Creator<BaseElement::NodeFactory, ObjectElement> SolverNodeClass("Solver");
Creator<BaseElement::NodeFactory, ObjectElement> CollisionNodeClass("Collision");
Creator<BaseElement::NodeFactory, ObjectElement> VisualNodeClass("Visual");
Creator<BaseElement::NodeFactory, ObjectElement> BehaviorNodeClass("Behavior");

const char* ObjectElement::getClass() const
{
    return ObjectNodeClass.c_str();
}

} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

