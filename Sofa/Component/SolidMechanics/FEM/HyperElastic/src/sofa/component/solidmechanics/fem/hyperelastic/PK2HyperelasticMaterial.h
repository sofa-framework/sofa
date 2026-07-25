#pragma once
#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticMaterial.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

/**
 * A hyperelastic material defined by its second Piola-Kirchhoff stress tensor and its Lagrangian
 * elasticity tensor.
 */
template <class TDataTypes>
class PK2HyperelasticMaterial : public HyperelasticMaterial<TDataTypes>
{
public:
    SOFA_CLASS(PK2HyperelasticMaterial<TDataTypes>, HyperelasticMaterial<TDataTypes>);
    using DataTypes = TDataTypes;

protected:
    using DeformationGradient = HyperelasticMaterial<TDataTypes>::DeformationGradient;
    using RightCauchyGreenTensor = HyperelasticMaterial<TDataTypes>::RightCauchyGreenTensor;
    using StressTensor = HyperelasticMaterial<TDataTypes>::StressTensor;
    using ElasticityTensor = HyperelasticMaterial<TDataTypes>::ElasticityTensor;
    using TangentModulus = HyperelasticMaterial<TDataTypes>::TangentModulus;
    using HyperelasticMaterial<TDataTypes>::spatial_dimensions;

public:
    StressTensor firstPiolaKirchhoffStress(Strain<DataTypes>& strain) final;
    TangentModulus materialTangentModulus(Strain<DataTypes>& strain) final;

protected:
    virtual StressTensor secondPiolaKirchhoffStress(Strain<DataTypes>& strain) = 0;
    virtual ElasticityTensor elasticityTensor(Strain<DataTypes>& strain) = 0;
};

#if !defined(ELASTICITY_COMPONENT_HYPERELASTIC_MATERIAL_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec3Types>;
#endif

}  // namespace elasticity
