#define ELASTICITY_COMPONENT_ELEMENT_LINEAR_SMALL_STRAIN_FEM_FORCE_FIELD_CPP

#include <sofa/component/solidmechanics/fem/elastic/ElementLinearSmallStrainFEMForceField.inl>

#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::fem::elastic
{

void registerElementLinearSmallStrainFEMForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear beams assuming small strain")
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear triangles assuming small strain")
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear quads assuming small strain")
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear tetrahedra assuming small strain")
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear hexahedra assuming small strain")
        .add< ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >(true));
}

template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}
