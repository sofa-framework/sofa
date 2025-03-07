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

#include <sofa/component/odesolver/backward/config.h>
#include <sofa/component/odesolver/backward/BaseLinearMultiStepMethod.h>

namespace sofa::component::odesolver::backward
{

struct SOFA_COMPONENT_ODESOLVER_BACKWARD_API BDF2Parameters
{
    static constexpr std::size_t Order = 2;
    static constexpr sofa::type::fixed_array<SReal, Order + 1> a_coef { 1 / 3_sreal, -4 / 3_sreal, 1_sreal};
    static constexpr sofa::type::fixed_array<SReal, Order + 1> b_coef {0, 0, 2 / 3_sreal};
};

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API BDF2OdeSolver :
public BaseLinearMultiStepMethod<BDF2Parameters>
{
public:
    SOFA_CLASS(BDF2OdeSolver, BaseLinearMultiStepMethod<BDF2Parameters>);
};

}
