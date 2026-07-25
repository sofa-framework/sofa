#pragma once

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/NeoHookeanMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
auto NeoHookeanMaterial<DataTypes>::secondPiolaKirchhoffStress(Strain<DataTypes>& strain) -> StressTensor
{
    static constexpr auto& I = Strain<DataTypes>::identity;
    const auto& C = strain.getRightCauchyGreenTensor();

    const DeformationGradient C_1 = sofa::type::inverse(C);
    const Real J = strain.getDeterminantDeformationGradient();

    return m_mu * (I - C_1) + m_lambda * std::log(J) * C_1;
}

template <class DataTypes>
auto NeoHookeanMaterial<DataTypes>::elasticityTensor(Strain<DataTypes>& strain) -> ElasticityTensor
{
    const auto& C = strain.getRightCauchyGreenTensor();
    const RightCauchyGreenTensor C_1 = sofa::type::inverse(C);
    const Real J = strain.getDeterminantDeformationGradient();
    const Real logJ = std::log(J);

    return ElasticityTensor(
        [mu = m_mu, lambda = m_lambda, &C_1, logJ](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return (mu - lambda * logJ) * (C_1(i, k) * C_1(l, j) + C_1(i, l) * C_1(k, j))
                + lambda * C_1(l, k) * C_1(i, j);
        });
}

}  // namespace elasticity
