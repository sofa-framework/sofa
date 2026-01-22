#pragma once

#include <sofa/component/solidmechanics/fem/elastic/BaseElementLinearFEMForceField.h>
#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/trait.h>

#include <sofa/component/solidmechanics/fem/elastic/FEMForceField.h>

#if !defined(ELASTICITY_COMPONENT_ELEMENT_LINEAR_SMALL_STRAIN_FEM_FORCE_FIELD_CPP)
#include <sofa/component/solidmechanics/fem/elastic/finiteelement/FiniteElement[all].h>
#endif

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes, class ElementType>
class ElementLinearSmallStrainFEMForceField :
    public BaseElementLinearFEMForceField<DataTypes, ElementType>,
    public FEMForceField<DataTypes, ElementType>
{
public:
    SOFA_CLASS2(
        SOFA_TEMPLATE2(ElementLinearSmallStrainFEMForceField, DataTypes, ElementType),
            SOFA_TEMPLATE2(BaseElementLinearFEMForceField, DataTypes, ElementType),
            SOFA_TEMPLATE2(FEMForceField, DataTypes, ElementType));

    /**
     * The purpose of this function is to register the name of this class according to the provided
     * pattern.
     *
     * Example: ElementLinearSmallStrainFEMForceField<Vec3Types, sofa::geometry::Edge> will produce
     * the class name "EdgeLinearSmallStrainFEMForceField".
     */
    static const std::string GetCustomClassName()
    {
        return std::string(sofa::geometry::elementTypeToString(ElementType::Element_type)) +
               "LinearSmallStrainFEMForceField";
    }

    static const std::string GetCustomTemplateName() { return DataTypes::Name(); }

private:
    using trait = sofa::component::solidmechanics::fem::elastic::trait<DataTypes, ElementType>;
    using ElementStiffness = typename trait::ElementStiffness;
    using ElasticityTensor = typename trait::ElasticityTensor;
    using ElementDisplacement = typename trait::ElementDisplacement;
    using StrainDisplacement = typename trait::StrainDisplacement;
    using ElementForce = typename trait::ElementForce;

public:
    void init() override;

    void buildStiffnessMatrix(sofa::core::behavior::StiffnessMatrix* matrix) override;

    SReal getPotentialEnergy(const sofa::core::MechanicalParams*, const sofa::DataVecCoord_t<DataTypes>& x) const override;

    using sofa::core::behavior::ForceField<DataTypes>::addKToMatrix;
    // almost deprecated, but here for compatibility with unit tests
    void addKToMatrix(sofa::linearalgebra::BaseMatrix* matrix, SReal kFact, unsigned& offset) override;

protected:

    void computeElementsForces(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& elementForces,
        const sofa::VecCoord_t<DataTypes>& nodePositions) override;

    void computeElementsForcesDeriv(
        const sofa::simulation::Range<std::size_t>& range,
        const sofa::core::MechanicalParams* mparams,
        sofa::type::vector<ElementForce>& elementForcesDeriv,
        const sofa::VecDeriv_t<DataTypes>& nodeDx) override;

};


#if !defined(ELASTICITY_COMPONENT_ELEMENT_LINEAR_SMALL_STRAIN_FEM_FORCE_FIELD_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec1Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Edge>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Triangle>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec2Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Quad>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Tetrahedron>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_ELASTIC_API ElementLinearSmallStrainFEMForceField<sofa::defaulttype::Vec3Types, sofa::geometry::Hexahedron>;
#endif

}  // namespace sofa::component::solidmechanics::fem::elastic
