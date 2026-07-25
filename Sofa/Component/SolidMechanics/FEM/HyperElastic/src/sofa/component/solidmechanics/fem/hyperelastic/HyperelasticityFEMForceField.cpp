#define ELASTICITY_COMPONENT_HYPERLASTICITY_FEM_FORCE_FIELD_CPP

#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticityFEMForceField.inl>

#include <sofa/fem/FiniteElement[all].h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

void registerHyperelasticityFEMForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Hyperelasticity")
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >()
        .add< HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >()
    );
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticityFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}
