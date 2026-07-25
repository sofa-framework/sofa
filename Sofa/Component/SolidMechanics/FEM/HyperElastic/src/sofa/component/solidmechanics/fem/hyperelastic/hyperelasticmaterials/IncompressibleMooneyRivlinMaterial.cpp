#define ELASTICITY_COMPONENT_MATERIAL_INCOMPRESSIBLEMOONEYRIVLIN_CPP

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/IncompressibleMooneyRivlinMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerIncompressibleMooneyRivlinMaterial(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Incompressible Mooney-Rivlin material")
        .add< IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec1Types> >()
        .add< IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec2Types> >()
        .add< IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec3Types> >(true));
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec3Types>;

}
