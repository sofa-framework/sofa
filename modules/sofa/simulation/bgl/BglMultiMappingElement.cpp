#include <sofa/simulation/bgl/BglMultiMappingElement.h>
#include <sofa/simulation/common/Node.h>
namespace sofa
{
namespace simulation
{
namespace bgl
{

using namespace sofa::simulation::xml;

BglMultiMappingElement::BglMultiMappingElement(const std::string& name, const std::string& type, BaseElement* parent/* =NULL */)
    :BaseMultiMappingElement(name,type,parent)
{

}

void BglMultiMappingElement::updateSceneGraph(sofa::core::BaseMapping* multiMapping,
        const helper::vector<simulation::Node*>& /*ancestorInputs*/,
        helper::vector<simulation::Node*>& otherInputs,
        helper::vector<simulation::Node*>& outputs)
{
    sofa::simulation::Node* currentNode = dynamic_cast<sofa::simulation::Node*>(multiMapping->getContext());

    helper::vector<sofa::simulation::Node*>::iterator iterNode;

    /* add the currentNode to the filteredInputNodes child list */
    for( iterNode = otherInputs.begin(); iterNode != otherInputs.end(); ++iterNode )
    {
        (*iterNode)->addChild(currentNode);
    }

    /* add the outputNodes to the currentNode child list */
    for( iterNode = outputs.begin(); iterNode != outputs.end(); ++iterNode)
    {
        if( *iterNode != currentNode)
        {
            currentNode->addChild(*iterNode);
        }
    }

}

SOFA_DECL_CLASS(BglMultiMappingElement)

helper::Creator<sofa::simulation::xml::BaseElement::NodeFactory, BglMultiMappingElement> BglNodeMultiMappingClass("BglNodeMultiMapping");

const char* BglMultiMappingElement::getClass() const
{
    return BglNodeMultiMappingClass.c_str();
}

}

}

}
