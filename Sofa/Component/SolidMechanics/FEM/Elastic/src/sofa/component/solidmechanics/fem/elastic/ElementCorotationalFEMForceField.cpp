#define ELASTICITY_COMPONENT_ELEMENT_COROTATIONAL_FEM_FORCE_FIELD_CPP

#include <sofa/component/solidmechanics/fem/elastic/ElementCorotationalFEMForceField.inl>

#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::solidmechanics::fem::elastic
{

void registerElementCorotationalFEMForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear beams using the corotational approach")
    //     .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge> >()
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge> >()
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear triangles using the corotational approach")
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle> >()
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear quads using the corotational approach")
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad> >()
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear tetrahedra using the corotational approach")
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron> >(true));

    factory->registerObjects(sofa::core::ObjectRegistrationData("Hooke's law on linear hexahedra using the corotational approach")
        .add< ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron> >(true));
}

// template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementCorotationalFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;

}
