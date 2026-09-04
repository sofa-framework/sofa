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

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_MATERIAL_OGDEN_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template <class DataTypes>
class OgdenMaterial :
    public PK2HyperelasticMaterial<DataTypes>
{
public:
    SOFA_CLASS(OgdenMaterial, PK2HyperelasticMaterial<DataTypes>);

private:
    using Real = sofa::Real_t<DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    using DeformationGradient = typename PK2HyperelasticMaterial<DataTypes>::DeformationGradient;
    using RightCauchyGreenTensor = typename PK2HyperelasticMaterial<DataTypes>::RightCauchyGreenTensor;
    using StressTensor = typename PK2HyperelasticMaterial<DataTypes>::StressTensor;
    using ElasticityTensor = typename PK2HyperelasticMaterial<DataTypes>::ElasticityTensor;

public:
    // Ogden material constants
    sofa::Data<Real> m_mu;
    sofa::Data<Real> m_alpha;
    sofa::Data<Real> m_kappa;

    StressTensor secondPiolaKirchhoffStress(Strain<DataTypes>& strain) override;

    ElasticityTensor elasticityTensor(Strain<DataTypes>& strain) override;

protected:
    OgdenMaterial();
};


#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_MATERIAL_OGDEN_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API OgdenMaterial<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API OgdenMaterial<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API OgdenMaterial<sofa::defaulttype::Vec3Types>;
#endif
}
