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

#include <sofa/component/solidmechanics/fem/elastic/config.h>
#include <sofa/core/objectmodel/BaseComponent.h>

#include <sofa/type/MatSym.h>

namespace sofa::component::solidmechanics::fem::elastic
{

template<class DataTypes>
class CauchyStressEvaluator : public virtual core::objectmodel::BaseComponent
{
public:
    SOFA_ABSTRACT_CLASS(CauchyStressEvaluator, BaseComponent)

    static constexpr sofa::Size spatial_dimensions = DataTypes::spatial_dimensions;
    using StressVoigtVector = sofa::type::Vec<sofa::type::NumberOfIndependentElements<spatial_dimensions>, sofa::Real_t<DataTypes>>;
    using DeformationGradient = sofa::type::Mat<spatial_dimensions, spatial_dimensions, Real_t<DataTypes>>;

    virtual StressVoigtVector computeStress(const DeformationGradient& F, sofa::Size elementId) = 0;
};

}
