#define ELASTICITY_COMPONENT_PK2HYPERELASTIC_MATERIAL_CPP
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec3Types>;

}
