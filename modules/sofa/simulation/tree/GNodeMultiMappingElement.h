#ifndef SOFA_SIMULATION_TREE_GNODEMULTIMAPPINGELEMENT_H
#define SOFA_SIMULATION_TREE_GNODEMULTIMAPPINGELEMENT_H

#include <sofa/simulation/common/xml/BaseMultiMappingElement.h>

namespace sofa
{

namespace simulation
{

namespace tree
{

class GNodeMultiMappingElement : public sofa::simulation::xml::BaseMultiMappingElement
{
public:
    GNodeMultiMappingElement(const std::string& name,
            const std::string& type,
            BaseElement* parent =NULL);

    const char* getClass() const;

protected:
    void updateSceneGraph(
        sofa::core::BaseMapping* multiMapping,
        const helper::vector<simulation::Node*>& ancestorInputs,
        helper::vector<simulation::Node*>& otherInputs,
        helper::vector<simulation::Node*>& outputs);
};



}

}

}


#endif // SOFA_SIMULATION_TREE_GNODEMULTIMAPPINGELEMENT_H
