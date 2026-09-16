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
#include <sofa/type/FullySymmetric4Tensor.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/impl/MajorSymmetric4Tensor.h>
#include <sofa/component/solidmechanics/fem/hyperelastic/impl/Strain.h>
#include <sofa/core/objectmodel/BaseObject.h>

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_HYPERELASTIC_MATERIAL_CPP)
#include <sofa/defaulttype/VecTypes.h>
#endif

namespace sofa::component::solidmechanics::fem::hyperelastic
{

template<class TDataTypes>
class HyperelasticMaterial : public virtual sofa::core::objectmodel::BaseObject
{
public:
    SOFA_ABSTRACT_CLASS(HyperelasticMaterial<TDataTypes>, sofa::core::objectmodel::BaseObject);
    using DataTypes = TDataTypes;

protected:
    using Real = sofa::Real_t<DataTypes>;

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;

    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;
    using RightCauchyGreenTensor = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;
    using StressTensor = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real>;
    using ElasticityTensor = sofa::type::FullySymmetric4Tensor<spatial_dimensions, Real>;
    using TangentModulus = MajorSymmetric4Tensor<DataTypes>;

public:
    void init() override;

    /**
     * Computes the First Piola-Kirchhoff stress tensor for a given deformation gradient.
     *
     * It corresponds to the derivative of the strain energy density function w.r.t. deformation
     * gradient.
     */
    virtual StressTensor firstPiolaKirchhoffStress(Strain<DataTypes>& strain) = 0;

    /**
     * Compute the jacobian of the first Piola-Kirchhoff stress tensor with respect to the
     * deformation gradient.
     *
     * It is called the material tangent modulus.
     */
    virtual TangentModulus materialTangentModulus(Strain<DataTypes>& strain) = 0;

};

#if !defined(SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_HYPERELASTIC_MATERIAL_CPP)
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticMaterial<sofa::defaulttype::Vec1Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticMaterial<sofa::defaulttype::Vec2Types>;
extern template class SOFA_COMPONENT_SOLIDMECHANICS_FEM_HYPERELASTIC_API HyperelasticMaterial<sofa::defaulttype::Vec3Types>;
#endif

}
