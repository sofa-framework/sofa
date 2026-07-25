#pragma once

#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.h>

#if !defined(ELASTICITY_COMPONENT_MATERIAL_INCOMPRESSIBLEMOONEYRIVLIN_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
class IncompressibleMooneyRivlinMaterial :
    public PK2HyperelasticMaterial<DataTypes>
{
public:
    SOFA_CLASS(IncompressibleMooneyRivlinMaterial, PK2HyperelasticMaterial<DataTypes>);

private:
    using Real = sofa::Real_t<DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    using DeformationGradient = PK2HyperelasticMaterial<DataTypes>::DeformationGradient;
    using RightCauchyGreenTensor = PK2HyperelasticMaterial<DataTypes>::RightCauchyGreenTensor;
    using StressTensor = PK2HyperelasticMaterial<DataTypes>::StressTensor;
    using ElasticityTensor = PK2HyperelasticMaterial<DataTypes>::ElasticityTensor;

public:
    sofa::Data<Real> m_mu10;
    sofa::Data<Real> m_mu01;

    StressTensor secondPiolaKirchhoffStress(Strain<DataTypes>& strain) override;

    ElasticityTensor elasticityTensor(Strain<DataTypes>& strain) override;

protected:
    IncompressibleMooneyRivlinMaterial();
};


#if !defined(ELASTICITY_COMPONENT_MATERIAL_INCOMPRESSIBLEMOONEYRIVLIN_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API IncompressibleMooneyRivlinMaterial<sofa::defaulttype::Vec3Types>;
#endif
}
