#define ELASTICITY_COMPONENT_MATERIAL_NEOHOOKEANMATERIAL_CPP

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/NeoHookeanMaterial.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerNeoHookeanMaterial(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Neo-Hookean material")
        .add< NeoHookeanMaterial<sofa::defaulttype::Vec1Types> >()
        .add< NeoHookeanMaterial<sofa::defaulttype::Vec2Types> >()
        .add< NeoHookeanMaterial<sofa::defaulttype::Vec3Types> >(true));
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API NeoHookeanMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API NeoHookeanMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API NeoHookeanMaterial<sofa::defaulttype::Vec3Types>;

}
