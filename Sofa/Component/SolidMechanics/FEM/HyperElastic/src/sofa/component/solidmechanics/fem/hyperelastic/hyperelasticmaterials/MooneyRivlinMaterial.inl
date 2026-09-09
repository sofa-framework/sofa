/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#pragma once

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/MooneyRivlinMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
MooneyRivlinMaterial<DataTypes>::MooneyRivlinMaterial()
    : m_mu10(initData(&m_mu10, static_cast<Real>(1e3), "mu10",
                      "Material constant associated to the first invariant"))
    , m_mu01(initData(&m_mu01, static_cast<Real>(1e3), "mu01",
                      "Material constant associated to the second invariant"))
    , m_bulkModulus(initData(&m_bulkModulus, static_cast<Real>(1e2), "bulkModulus", "Bulk modulus"))
{}

template <class DataTypes>
auto MooneyRivlinMaterial<DataTypes>::secondPiolaKirchhoffStress(Strain<DataTypes>& strain) -> StressTensor
{
    const auto& C = strain.getRightCauchyGreenTensor();
    const auto C_1 = sofa::type::inverse(C);

    const auto J = strain.getDeterminantDeformationGradient();
    assert(J > 0);

    const auto S_isochoric = [this, J, &strain, &C_1, &C]()
    {
        static constexpr auto& I = Strain<DataTypes>::identity;
        static constexpr Real dim_1 = static_cast<Real>(1) / static_cast<Real>(spatial_dimensions);

        const auto mu10 = m_mu10.getValue();
        const auto mu01 = m_mu01.getValue();

        const auto invariant1 = strain.getInvariant1();
        const auto invariant2 = strain.getInvariant2();

        const auto S_mu_10 = pow(J, -static_cast<Real>(2) * dim_1) * (I - dim_1 * invariant1 * C_1);
        const auto S_mu_01 = pow(J, -static_cast<Real>(4) * dim_1) * (invariant1 * I - C - static_cast<Real>(2) * dim_1 * invariant2 * C_1);

        return static_cast<Real>(2) * (mu10 * S_mu_10 + mu01 * S_mu_01);
    }();

    const auto S_volumetric = [this, J, &C_1]()
    {
        const auto bulk = m_bulkModulus.getValue();
        return bulk * log(J) * C_1;
    }();

    return S_isochoric + S_volumetric;
}

template <class DataTypes>
auto MooneyRivlinMaterial<DataTypes>::elasticityTensor(Strain<DataTypes>& strain) -> ElasticityTensor
{
    static constexpr Real dim_1 = static_cast<Real>(1) / static_cast<Real>(spatial_dimensions);
    const auto& C = strain.getRightCauchyGreenTensor();
    const auto J = strain.getDeterminantDeformationGradient();
    const auto logJ = log(J);
    const auto J_2dim = pow(J, -2 * dim_1);
    const auto J_4dim = pow(J, -4 * dim_1);
    const auto C_1 = sofa::type::inverse(C);
    const auto I1 = strain.getInvariant1();
    const auto I2 = strain.getInvariant2();

    const auto mu01 = m_mu01.getValue();
    const auto mu10 = m_mu10.getValue();
    const auto bulk = m_bulkModulus.getValue();

    auto delta = [](auto i, auto j){ return sofa::component::solidmechanics::fem::elastic::kroneckerDelta<Real>(i, j); };

    return ElasticityTensor(
        [&](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            //derivative of C^{-1} with respect to C
            const auto dC_1dC = -static_cast<Real>(0.5) * (C_1(i, k) * C_1(l, j) + C_1(i, l) * C_1(k, j));

            // the derivative of S_mu_10 with respect to C
            // each term has both minor and major symmetries
            const Real dS_mu_10dC = -dim_1 * J_2dim * (
                delta(i, j) * C_1(k, l) + delta(k, l) * C_1(i, j)
                - dim_1 * C_1(k, l) * I1 * C_1(i, j)
                + I1 * dC_1dC
            );

            // The derivative of S_mu_01 with respect to C
            // the terms have been grouped to highlight the minor and major symmetries
            const Real dS_mu_01dC = J_4dim * (
                - 2 * dim_1 * (
                    C_1(k, l) * (I1 * delta(i, j) - C(i, j)) +
                    C_1(i, j) * (I1 * delta(k, l) - C(k, l))
                    + dC_1dC * I2
                )
                + delta(i, j) * delta(k, l)
                - static_cast<Real>(0.5) * (
                        delta(i, k) * delta(j, l) +
                        delta(i, l) * delta(j, k))
                + 4 * dim_1 * dim_1 * C_1(k, l) * C_1(i, j) * I2
            );

            const Real dS_isochoric_dC = 2 * (mu10 * dS_mu_10dC + mu01 * dS_mu_01dC);

            // the derivative of S_volumetric with respect to C
            // this term has both minor and major symmetries
            const Real dS_volumetric_dC = bulk * (C_1(l, k) * C_1(i, j) / 2 + logJ * dC_1dC);

            return 2 * (dS_isochoric_dC + dS_volumetric_dC);
        });
}

}  // namespace elasticity
