#define ELASTICITY_COMPONENT_MATERIAL_OGDEN_CPP

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/OgdenMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerOgdenMaterial(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Ogden material")
        .add< OgdenMaterial<sofa::defaulttype::Vec1Types> >()
        .add< OgdenMaterial<sofa::defaulttype::Vec2Types> >()
        .add< OgdenMaterial<sofa::defaulttype::Vec3Types> >(true));
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API OgdenMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API OgdenMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API OgdenMaterial<sofa::defaulttype::Vec3Types>;

}
