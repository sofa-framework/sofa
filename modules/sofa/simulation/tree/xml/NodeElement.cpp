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

NodeElement::NodeElement(const std::string& name, const std::string& type, BaseElement* parent)
    : Element<core::objectmodel::BaseNode>(name, type, parent)
{
}

NodeElement::~NodeElement()
{
}

bool NodeElement::setParent(BaseElement* newParent)
{
    if (newParent != NULL && dynamic_cast<NodeElement*>(newParent)==NULL)
        return false;
    else
        return Element<core::objectmodel::BaseNode>::setParent(newParent);
}

bool NodeElement::initNode()
{
    if (!Element<core::objectmodel::BaseNode>::initNode()) return false;
    if (getObject()!=NULL && getParent()!=NULL && dynamic_cast<core::objectmodel::BaseNode*>(getParent()->getBaseObject())!=NULL)
    {
        std::cout << "Adding Child "<<getName()<<" to "<<getParent()->getName()<<std::endl;
        dynamic_cast<core::objectmodel::BaseNode*>(getParent()->getBaseObject())->addChild(getObject());
    }
    return true;
}

bool NodeElement::init()
{
    bool res = Element<core::objectmodel::BaseNode>::init();
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

helper::Creator<BaseElement::NodeFactory, NodeElement> NodeNodeClass("Node");
helper::Creator<BaseElement::NodeFactory, NodeElement> NodeBodyClass("Body");
helper::Creator<BaseElement::NodeFactory, NodeElement> NodeGClass("G");

const char* NodeElement::getClass() const
{
    return NodeNodeClass.c_str();
}

} // namespace xml

} // namespace tree

} // namespace simulation

} // namespace sofa

