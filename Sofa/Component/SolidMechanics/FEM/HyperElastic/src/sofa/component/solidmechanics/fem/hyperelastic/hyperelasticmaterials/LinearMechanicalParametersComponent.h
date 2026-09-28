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
#include <sofa/core/objectmodel/BaseObject.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_LINEAR_MECHANICAL_PARAMETERS_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template<class DataTypes>
class LinearMechanicalParametersComponent : public virtual sofa::core::objectmodel::BaseObject
{
    using Real = sofa::Real_t<DataTypes>;

public:
    SOFA_CLASS(LinearMechanicalParametersComponent<DataTypes>, sofa::core::objectmodel::BaseObject);

    sofa::Data<Real> d_poissonRatio;
    sofa::Data<Real> d_youngModulus;

protected:
    LinearMechanicalParametersComponent();

    void setLameCoefficients();

    // Lamé's coefficients
    Real m_lambda, m_mu;
};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_LINEAR_MECHANICAL_PARAMETERS_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API LinearMechanicalParametersComponent<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API LinearMechanicalParametersComponent<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API LinearMechanicalParametersComponent<sofa::defaulttype::Vec3Types>;
#endif
}  // namespace elasticity
