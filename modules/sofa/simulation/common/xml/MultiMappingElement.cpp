#include <sofa/simulation/common/xml/MultiMappingElement.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/BaseState.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/BaseNode.h>
#include <sofa/simulation/common/xml/NodeElement.h>
#include <sofa/simulation/common/xml/Element.inl>

namespace sofa
{
namespace simulation
{
namespace xml
{
MultiMappingElement::MultiMappingElement(const std::string& name, const std::string& type, BaseElement* parent/* =NULL */)
    :ObjectElement(name,type,parent)
{

}

bool MultiMappingElement::initNode()
{
    using namespace core::objectmodel;
    using namespace core;
    bool result = ObjectElement::initNode();


    if( result )
    {

        BaseMapping* multimapping =  dynamic_cast<BaseMapping*>(this->getTypedObject());
        NodeElement* currentNodeElement = dynamic_cast<NodeElement *>(getParent());
        simulation::Node* currentNode =  dynamic_cast<simulation::Node* >( currentNodeElement->getTypedObject() );
        helper::vector<core::BaseState*> inputStates  = multimapping->getFrom();
        helper::vector<core::BaseState*> outputStates = multimapping->getTo();


        helper::vector<core::BaseState*>::iterator iterState;
        helper::vector<simulation::Node*> inputNodes, outputNodes;

        /* get the Nodes corresponding to each input BaseState context */
        for( iterState = inputStates.begin();  iterState != inputStates.end(); ++iterState)
        {
            simulation::Node* node = dynamic_cast< simulation::Node* >((*iterState)->getContext());
            inputNodes.push_back(node);
        }
        /* */
        /* get the Nodes corresponding to each output BaseState context */
        for( iterState = outputStates.begin(); iterState != outputStates.end(); ++iterState)
        {
            simulation::Node* node = dynamic_cast< simulation::Node* >((*iterState)->getContext());
            outputNodes.push_back(node);
        }

        helper::vector<simulation::Node*>::iterator iterNode;
        BaseNode* currentBaseNode;

        /* filter out inputNodes which already belong to the currentNode ancestors */
        helper::vector<simulation::Node*> filteredInputNodes;
        iterNode = inputNodes.begin();
        currentBaseNode = currentNode;
        for( iterNode = inputNodes.begin(); iterNode != inputNodes.end(); ++iterNode)
        {
            if( !currentBaseNode->hasAncestor(*iterNode) )
            {
                filteredInputNodes.push_back(*iterNode);
            }
        }

        /* add the currentNode to the filteredInputNodes child list */
        for( iterNode = filteredInputNodes.begin(); iterNode != filteredInputNodes.end(); ++iterNode )
        {
            (*iterNode)->addChild(currentNode);
        }

        /* add the outputNodes to the currentNode child list */
        for( iterNode = outputNodes.begin(); iterNode != outputNodes.end(); ++iterNode)
        {
            if( *iterNode != currentNode)
            {
                currentNode->addChild(*iterNode);
            }
        }
    }

    return result;
}

SOFA_DECL_CLASS(MultiMappingObject)

Creator<BaseElement::NodeFactory, MultiMappingElement> MultiMappingNodeClass("MultiMappingObject");

const char* MultiMappingElement::getClass() const
{
    return MultiMappingNodeClass.c_str();
}


}
}
}
