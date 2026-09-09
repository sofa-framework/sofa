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
    using DeformationGradient = typename HyperelasticMaterial<TDataTypes>::DeformationGradient;
    using RightCauchyGreenTensor = typename HyperelasticMaterial<TDataTypes>::RightCauchyGreenTensor;
    using StressTensor = typename HyperelasticMaterial<TDataTypes>::StressTensor;
    using ElasticityTensor = typename HyperelasticMaterial<TDataTypes>::ElasticityTensor;
    using TangentModulus = typename HyperelasticMaterial<TDataTypes>::TangentModulus;
    using HyperelasticMaterial<TDataTypes>::spatial_dimensions;

public:
    StressTensor firstPiolaKirchhoffStress(Strain<DataTypes>& strain) final;
    TangentModulus materialTangentModulus(Strain<DataTypes>& strain) final;

protected:
    virtual StressTensor secondPiolaKirchhoffStress(Strain<DataTypes>& strain) = 0;
    virtual ElasticityTensor elasticityTensor(Strain<DataTypes>& strain) = 0;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_PK2HYPERELASTIC_MATERIAL_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API PK2HyperelasticMaterial<sofa::defaulttype::Vec3Types>;
#endif

}  // namespace elasticity
