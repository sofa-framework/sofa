#pragma once
#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticMaterial.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
void HyperelasticMaterial<DataTypes>::init()
{
    sofa::core::objectmodel::BaseObject::init();

    if (!this->isComponentStateInvalid())
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

}  // namespace elasticity
