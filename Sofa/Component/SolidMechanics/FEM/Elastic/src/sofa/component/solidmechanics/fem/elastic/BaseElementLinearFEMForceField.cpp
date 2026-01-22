#define ELASTICITY_COMPONENT_BASE_ELEMENT_LINEAR_FEM_FORCEFIELD_CPP
#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.inl>
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>

namespace sofa::component::solidmechanics::fem::elastic
{
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
}
