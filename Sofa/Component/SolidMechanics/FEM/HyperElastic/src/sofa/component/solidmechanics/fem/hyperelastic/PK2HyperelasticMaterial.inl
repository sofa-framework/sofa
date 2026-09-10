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

#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/KroneckerDelta.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/HyperelasticMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class TDataTypes>
auto PK2HyperelasticMaterial<TDataTypes>::firstPiolaKirchhoffStress(Strain<DataTypes>& strain) -> StressTensor
{
    const auto& F = strain.deformationGradient();
    const auto S = secondPiolaKirchhoffStress(strain);
    return F * S;
}

template <class TDataTypes>
auto PK2HyperelasticMaterial<TDataTypes>::materialTangentModulus(Strain<DataTypes>& strain) -> TangentModulus
{
    using Real = sofa::Real_t<TDataTypes>;
    SCOPED_TIMER_TR("tangentModulus");

    const auto& F = strain.deformationGradient();
    const auto C = elasticityTensor(strain);
    const auto S = secondPiolaKirchhoffStress(strain);

    const auto A = TangentModulus([&F, &C, &S](sofa::Index i, sofa::Index j, sofa::Index k, sofa::Index l)
    {
        auto A_ijkl = sofa::component::solidmechanics::fem::elastic::kroneckerDelta<Real>(i,k) * S(l, j);
        for (std::size_t q = 0; q < spatial_dimensions; ++q)
        {
            for (std::size_t r = 0; r < spatial_dimensions; ++r)
            {
                A_ijkl += F(i, q) * C(q, j, l, r) * F(k, r);
            }
        }
        return A_ijkl;
    });

    return A;
}

}  // namespace elasticity
