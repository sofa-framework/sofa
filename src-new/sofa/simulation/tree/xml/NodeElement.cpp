#include "NodeNode.h"
#include "ObjectNode.h"
#include "Node.inl"

namespace Sofa
{

namespace Components
{

namespace XML
{

using namespace Common;

//template class Factory< std::string, Abstract::BaseNode, Node<Abstract::BaseNode*>* >;

NodeNode::NodeNode(const std::string& name, const std::string& type, BaseNode* parent)
    : Node<Abstract::BaseNode>(name, type, parent)
{
}

NodeNode::~NodeNode()
{
}

bool NodeNode::setParent(BaseNode* newParent)
{
    if (newParent != NULL && dynamic_cast<NodeNode*>(newParent)==NULL)
        return false;
    else
        return Node<Abstract::BaseNode>::setParent(newParent);
}

bool NodeNode::initNode()
{
    if (!Node<Abstract::BaseNode>::initNode()) return false;
    if (getObject()!=NULL && getParent()!=NULL && dynamic_cast<Abstract::BaseNode*>(getParent()->getBaseObject())!=NULL)
    {
        std::cout << "Adding Child "<<getName()<<" to "<<getParent()->getName()<<std::endl;
        dynamic_cast<Abstract::BaseNode*>(getParent()->getBaseObject())->addChild(getObject());
    }
    return true;
}

bool NodeNode::init()
{
    bool res = Node<Abstract::BaseNode>::init();
    /*
    if (getObject()!=NULL)
    {
    	for (child_iterator<> it = begin();
    				it != end(); ++it)
    	{
    		Abstract::BaseObject* obj = dynamic_cast<Abstract::BaseObject*>(it->getBaseObject());
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

Creator<BaseNode::NodeFactory, NodeNode> NodeNodeClass("Node");
Creator<BaseNode::NodeFactory, NodeNode> NodeBodyClass("Body");
Creator<BaseNode::NodeFactory, NodeNode> NodeGClass("G");

const char* NodeNode::getClass() const
{
    return NodeNodeClass.c_str();
}

} // namespace XML

} // namespace Components

} // namespace Sofa
