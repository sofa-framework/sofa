#define ELASTICITY_COMPONENT_MATERIAL_MOONEYRIVLIN_CPP

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/MooneyRivlinMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerMooneyRivlinMaterial(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Mooney-Rivlin material")
        .add< MooneyRivlinMaterial<sofa::defaulttype::Vec1Types> >()
        .add< MooneyRivlinMaterial<sofa::defaulttype::Vec2Types> >()
        .add< MooneyRivlinMaterial<sofa::defaulttype::Vec3Types> >(true));
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API MooneyRivlinMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API MooneyRivlinMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API MooneyRivlinMaterial<sofa::defaulttype::Vec3Types>;

}
