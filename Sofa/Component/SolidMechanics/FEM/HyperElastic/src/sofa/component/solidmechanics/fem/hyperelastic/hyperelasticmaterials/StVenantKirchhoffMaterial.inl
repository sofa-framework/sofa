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

#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/StVenantKirchhoffMaterial.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/OrthotropicElasticityTensor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.inl>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
auto StVenantKirchhoffMaterial<DataTypes>::secondPiolaKirchhoffStress(Strain<DataTypes>& strain)
-> StressTensor
{
    static const auto& I = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>::Identity();

    // Green-Lagrangian strain tensor
    const auto& E = strain.getGreenLagrangeTensor();

    // Second Piola-Kirchhoff stress tensor
    return m_lambda * sofa::type::trace(E) * I + static_cast<Real>(2) * m_mu * E;
}

template <class DataTypes>
auto StVenantKirchhoffMaterial<DataTypes>::elasticityTensor(Strain<DataTypes>& strain) -> ElasticityTensor
{
    SCOPED_TIMER_TR("elasticityTensor");
    SOFA_UNUSED(strain);

    sofa::component::solidmechanics::fem::elastic::LameLambda<Real> lambda { m_lambda };
    sofa::component::solidmechanics::fem::elastic::LameMu<Real> mu { m_mu };

    return sofa::component::solidmechanics::fem::elastic::makeIsotropicElasticityTensor<spatial_dimensions>(mu, lambda);
}

}  // namespace elasticity
