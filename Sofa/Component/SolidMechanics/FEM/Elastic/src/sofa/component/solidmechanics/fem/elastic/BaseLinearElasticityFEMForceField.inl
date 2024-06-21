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
#include <sofa/component/solidmechanics/fem/elastic/BaseLinearElasticityFEMForceField.h>
#include <sofa/core/behavior/ForceField.inl>

namespace sofa::component::solidmechanics::fem::elastic
{

template <class DataTypes>
BaseLinearElasticityFEMForceField<DataTypes>::BaseLinearElasticityFEMForceField()
    : d_poissonRatio(initData(&d_poissonRatio,(Real)0.45,"poissonRatio","FEM Poisson Ratio in Hooke's law [0,0.5["))
    , d_youngModulus(initData(&d_youngModulus, defaultYoungModulusValue, "youngModulus","FEM Young's Modulus in Hooke's law"))
    , l_topology(initLink("topology", "link to the topology container"))
{
    d_poissonRatio.setRequired(true);
    d_poissonRatio.setWidget("poissonRatio");

    d_youngModulus.setRequired(true);
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::setPoissonRatio(Real val)
{
    this->d_poissonRatio.setValue(val);
}

template <class DataTypes>
void BaseLinearElasticityFEMForceField<DataTypes>::setYoungModulus(Real val)
{
    VecReal newY;
    newY.resize(1);
    newY[0] = val;
    d_youngModulus.setValue(newY);
}

template <class DataTypes>
typename BaseLinearElasticityFEMForceField<DataTypes>::Real
BaseLinearElasticityFEMForceField<DataTypes>::getYoungModulusInElement(sofa::Size elementId)
{
    Real youngModulusElement {};

    const auto& youngModulus = d_youngModulus.getValue();
    if (youngModulus.size() > elementId)
    {
        youngModulusElement = youngModulus[elementId];
    }
    else if (youngModulus.size() > 0)
    {
        youngModulusElement = youngModulus[0];
    }
    else
    {
        setYoungModulus(5000);
        youngModulusElement = youngModulus[0];
    }
    return youngModulusElement;
}

}
