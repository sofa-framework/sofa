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

#include <sofa/component/solidmechanics/fem/hyperelastic/config.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/PK2HyperelasticMaterial.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/hyperelasticmaterials/LinearMechanicalParametersComponent.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_MATERIAL_ST_VENANT_KIRCHHOFF_MATERIAL_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::hyperelastic
{

/**
 * @class StVenantKirchhoffMaterial
 * @brief Represents the St. Venant-Kirchhoff material model for hyperelastic materials.
 *
 * The St. Venant-Kirchhoff material is a simple model used to describe the stress-strain behavior
 * of isotropic hyperelastic materials. It is based on the linear elastic theory but is extended
 * to handle large deformations using the Green-Lagrange strain tensor and the Second
 * Piola-Kirchhoff stress tensor.
 *
 * This material model is only valid for cases where the deformation is relatively small, despite
 * being nonlinear in nature. Its application is mainly limited due to its inability to correctly
 * predict behavior under significant strain, as it does not accurately represent material
 * nonlinearity under large deformation.
 *
 * The material model is defined using two parameters:
 * - The Young's modulus: Describes the material's stiffness.
 * - The Poisson's ratio: Represents the material's ability to undergo deformation in directions
 *   orthogonal to the applied stress.
 */
template <class DataTypes>
class StVenantKirchhoffMaterial:
    public PK2HyperelasticMaterial<DataTypes>,
    public LinearMechanicalParametersComponent<DataTypes>
{
public:
    SOFA_CLASS2(StVenantKirchhoffMaterial, PK2HyperelasticMaterial<DataTypes>, LinearMechanicalParametersComponent<DataTypes>);

private:
    using Real = sofa::Real_t<DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    using DeformationGradient = typename PK2HyperelasticMaterial<DataTypes>::DeformationGradient;
    using RightCauchyGreenTensor = typename PK2HyperelasticMaterial<DataTypes>::RightCauchyGreenTensor;
    using StressTensor = typename PK2HyperelasticMaterial<DataTypes>::StressTensor;
    using ElasticityTensor = typename PK2HyperelasticMaterial<DataTypes>::ElasticityTensor;

    using LinearMechanicalParametersComponent<DataTypes>::m_lambda;
    using LinearMechanicalParametersComponent<DataTypes>::m_mu;

protected:
    StressTensor secondPiolaKirchhoffStress(Strain<DataTypes>& strain) override;

    ElasticityTensor elasticityTensor(Strain<DataTypes>& strain) override;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_MATERIAL_ST_VENANT_KIRCHHOFF_MATERIAL_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StVenantKirchhoffMaterial<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StVenantKirchhoffMaterial<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API StVenantKirchhoffMaterial<sofa::defaulttype::Vec3Types>;
#endif
}
