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
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/LinearMechanicalParametersComponent.h>
#include <sofa/component/solidmechanics/fem/elastic/impl/LameParameters.h>

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
LinearMechanicalParametersComponent<DataTypes>::LinearMechanicalParametersComponent()
: d_poissonRatio(initData(&d_poissonRatio, static_cast<Real>(0.45), "poissonRatio",
    "Poisson's ratio: represents the material's ability to undergo deformation in directions orthogonal to the applied stress"))
, d_youngModulus(initData(&d_youngModulus, static_cast<Real>(1e6), "youngModulus",
    "Young's modulus: describes the material's stiffness"))
{
    this->addUpdateCallback("toLameCoefficients", {&this->d_youngModulus, &this->d_poissonRatio},
    [this](const sofa::core::DataTracker& )
    {
        setLameCoefficients();
        return this->getComponentState();
    }, {});
    setLameCoefficients();
}

template <class DataTypes>
void LinearMechanicalParametersComponent<DataTypes>::setLameCoefficients()
{
    sofa::component::solidmechanics::fem::elastic::LameLambda<Real> lambdaStrong { 0 };
    sofa::component::solidmechanics::fem::elastic::LameMu<Real> muStrong { 0 };

    sofa::component::solidmechanics::fem::elastic::toLameParameters<DataTypes::spatial_dimensions, Real>(
        sofa::component::solidmechanics::fem::elastic::YoungModulus<Real>(this->d_youngModulus.getValue()),
        sofa::component::solidmechanics::fem::elastic::PoissonRatio<Real>(this->d_poissonRatio.getValue()),
        lambdaStrong, muStrong);

    m_lambda = lambdaStrong.get();
    m_mu = muStrong.get();
}

}  // namespace elasticity
