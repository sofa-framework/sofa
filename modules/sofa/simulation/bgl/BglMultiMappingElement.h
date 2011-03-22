#ifndef SOFA_SIMULATION_BGL_BGLMUTLIMAPPINGELEMENT_H
#define SOFA_SIMULATION_BGL_BGLMUTLIMAPPINGELEMENT_H

#include <sofa/simulation/common/xml/BaseMultiMappingElement.h>

namespace sofa
{
namespace simulation
{
namespace bgl
{

class BglMultiMappingElement : public sofa::simulation::xml::BaseMultiMappingElement
{
public:
    BglMultiMappingElement(const std::string& name, const std::string& type, BaseElement* parent=NULL);
    const char* getClass() const;
protected:
    virtual void updateSceneGraph(
        sofa::core::BaseMapping* multiMapping,
        const helper::vector<simulation::Node*>& ancestorInputs,
        helper::vector<simulation::Node*>& otherInputs,
        helper::vector<simulation::Node*>& outputs);

};

} // namespace bgl

} // namespace simulation

} // namespace sofa

#endif // SOFA_SIMULATION_BGL_BGLMUTLIMAPPINGELEMENT_H
