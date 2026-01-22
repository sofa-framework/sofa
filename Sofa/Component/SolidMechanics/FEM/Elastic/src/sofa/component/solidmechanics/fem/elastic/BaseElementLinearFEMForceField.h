#pragma once

#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>

#if !defined(ELASTICITY_COMPONENT_BASE_ELEMENT_LINEAR_FEM_FORCEFIELD_CPP)
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * A base class for all element-based linear elastic force fields.
 *
 * It stores precomputed stiffness matrices (one per element) that are derived from:
 *   - The initial configuration of the mechanical model
 *   - Material properties (Young's modulus, Poisson's ratio)
 */
template <class DataTypes, class ElementType>
class BaseElementLinearFEMForceField : public sofa::component::solidmechanics::fem::elastic::BaseLinearElasticityFEMForceField<DataTypes>
{
public:
    SOFA_ABSTRACT_CLASS(
        SOFA_TEMPLATE2(BaseElementLinearFEMForceField, DataTypes, ElementType),
        sofa::component::solidmechanics::fem::elastic::BaseLinearElasticityFEMForceField<DataTypes>);

    void init() override;

private:
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    using ElementStiffness = typename trait::ElementStiffness;
    using ElasticityTensor = typename trait::ElasticityTensor;
    using StrainDisplacement = typename trait::StrainDisplacement;

protected:

    BaseElementLinearFEMForceField();

    /**
     * With linear small strain, the element stiffness matrix is constant, so it can be precomputed.
     */
    void precomputeElementStiffness();

public:

    /**
     * List of precomputed element stiffness matrices
     */
    sofa::Data<sofa::type::vector<ElementStiffness> > d_elementStiffness;
};

#if !defined(ELASTICITY_COMPONENT_BASE_ELEMENT_LINEAR_FEM_FORCEFIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API BaseElementLinearFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
