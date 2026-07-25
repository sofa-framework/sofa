#define ELASTICITY_COMPONENT_HYPERELASTIC_MATERIAL_CPP

#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticMaterial.inl>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticMaterial<sofa::defaulttype::Vec3Types>;

}
