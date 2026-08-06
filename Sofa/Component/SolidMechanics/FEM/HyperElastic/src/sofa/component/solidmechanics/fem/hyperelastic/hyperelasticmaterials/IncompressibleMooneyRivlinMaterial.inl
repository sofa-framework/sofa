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

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/IncompressibleMooneyRivlinMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
IncompressibleMooneyRivlinMaterial<DataTypes>::IncompressibleMooneyRivlinMaterial()
    : m_mu10(initData(&m_mu10, static_cast<Real>(1e3), "mu10",
                      "Material constant associated to the first invariant"))
    , m_mu01(initData(&m_mu01, static_cast<Real>(1e3), "mu01",
                      "Material constant associated to the second invariant"))
{}

template <class DataTypes>
auto IncompressibleMooneyRivlinMaterial<DataTypes>::secondPiolaKirchhoffStress(Strain<DataTypes>& strain) -> StressTensor
{
    static constexpr auto& I = Strain<DataTypes>::identity;
    const auto& C = strain.getRightCauchyGreenTensor();

    const auto invariant1 = strain.getInvariant1();

    const auto mu10 = m_mu10.getValue();
    const auto mu01 = m_mu01.getValue();

    return static_cast<Real>(2) * mu10 * I + static_cast<Real>(2) * mu01 * (invariant1 * I - C);
}

template <class DataTypes>
auto IncompressibleMooneyRivlinMaterial<DataTypes>::elasticityTensor(Strain<DataTypes>& strain) -> ElasticityTensor
{
    auto delta = [](auto i, auto j){ return sofa::component::solidmechanics::fem::elastic::kroneckerDelta<Real>(i, j); };

    return ElasticityTensor(
        [mu01 = m_mu01.getValue(), &delta](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
        {
            return 2 * mu01 * (
                2 * delta(i, j) * delta(k, l)
                - delta(i, k) * delta(j, l)
                - delta(i, l) * delta(j, k)
                );
        });
}

}  // namespace elasticity
