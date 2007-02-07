#include <sofa/simulation/tree/xml/NodeElement.h>
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

//template class Factory< std::string, xml::BaseElement, Node<xml::BaseElement*>* >;

NodeElement::NodeNode(const std::string& name, const std::string& type, BaseElement* parent)
    : Node<xml::BaseElement>(name, type, parent)
{
}

NodeElement::~NodeNode()
{
}

bool NodeElement::setParent(BaseElement* newParent)
{
    if (newParent != NULL && dynamic_cast<NodeNode*>(newParent)==NULL)
        return false;
    else
        return Node<xml::BaseElement>::setParent(newParent);
}

bool NodeElement::initNode()
{
    if (!Node<xml::BaseElement>::initNode()) return false;
    if (getObject()!=NULL && getParent()!=NULL && dynamic_cast<xml::BaseElement*>(getParent()->getBaseObject())!=NULL)
    {
        std::cout << "Adding Child "<<getName()<<" to "<<getParent()->getName()<<std::endl;
        dynamic_cast<xml::BaseElement*>(getParent()->getBaseObject())->addChild(getObject());
    }
    return true;
}

bool NodeElement::init()
{
    bool res = Node<xml::BaseElement>::init();
    /*
    if (getObject()!=NULL)
    {
    	for (child_iterator<> it = begin();
    				it != end(); ++it)
    	{
    		objectmodel::BaseObject* obj = dynamic_cast<objectmodel::BaseObject*>(it->getBaseObject());
    		if (obj!=NULL)
    		{
    			std::cout << "Adding Object "<<it->getName()<<" to "<<getName()<<std::endl;
    			getObject()->addObject(obj);
    		}
    	}
    }
    */
    return res;
}

SOFA_DECL_CLASS(Node)

Creator<BaseElement::NodeFactory, NodeNode> NodeNodeClass("Node");
Creator<BaseElement::NodeFactory, NodeNode> NodeBodyClass("Body");
Creator<BaseElement::NodeFactory, NodeNode> NodeGClass("G");

const char* NodeElement::getClass() const
{
    return NodeNodeClass.c_str();
}

} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

