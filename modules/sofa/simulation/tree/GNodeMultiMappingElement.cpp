#include <sofa/simulation/tree/GNodeMultiMappingElement.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/Factory.h>

namespace sofa
{

namespace simulation
{

namespace tree
{
GNodeMultiMappingElement::GNodeMultiMappingElement(const std::string &name,
        const std::string &type, BaseElement *parent /*= 0*/)
    :BaseMultiMappingElement(name,type,parent)
{

}

void GNodeMultiMappingElement::updateSceneGraph(
    sofa::core::BaseMapping* multiMapping,
    const helper::vector<simulation::Node*>& /*ancestorInputs*/,
    helper::vector<simulation::Node*>& otherInputs,
    helper::vector<simulation::Node*>& /*outputs*/)
{

    helper::vector<simulation::Node*>::const_iterator it;
    for( it = otherInputs.begin(); it != otherInputs.end(); ++it)
    {
        multiMapping->serr << "Node: " << (*it)->getName() << " does not belong to "
                << multiMapping->getContext()->getName() << "ancestors" << multiMapping->sendl;
    }
}


SOFA_DECL_CLASS(GNodeMultiMappingElement)

helper::Creator<sofa::simulation::xml::BaseElement::NodeFactory, GNodeMultiMappingElement> GNodeMultiMappingClass("GNodeMultiMapping");

const char* GNodeMultiMappingElement::getClass() const
{
    return GNodeMultiMappingClass.c_str();
}


} // namespace tree

} // namespace simulation

} // namespace sofa
