#define ELASTICITY_COMPONENT_MATERIAL_ST_VENANT_KIRCHHOFF_MATERIAL_CPP

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/StVenantKirchhoffMaterial.inl>

#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerStVenantKirchhoffMaterial(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Saint Venant-Kirchhoff material model for hyperelastic materials")
        .add< StVenantKirchhoffMaterial<sofa::defaulttype::Vec1Types> >()
        .add< StVenantKirchhoffMaterial<sofa::defaulttype::Vec2Types> >()
        .add< StVenantKirchhoffMaterial<sofa::defaulttype::Vec3Types> >(true));
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StVenantKirchhoffMaterial<sofa::defaulttype::Vec1Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StVenantKirchhoffMaterial<sofa::defaulttype::Vec2Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StVenantKirchhoffMaterial<sofa::defaulttype::Vec3Types>;

}
