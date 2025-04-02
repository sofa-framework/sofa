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

class SOFA_COMPONENT_ODESOLVER_BACKWARD_API BDFOdeSolver :
    public BaseLinearMultiStepMethod
{
public:
    SOFA_CLASS(BDFOdeSolver, BaseLinearMultiStepMethod);

    static void computeLinearMultiStepCoefficients(const std::deque<SReal>& samples,
        sofa::type::vector<SReal>& a_coef, sofa::type::vector<SReal>& b_coef);

protected:
    void recomputeCoefficients(std::size_t order, SReal dt) override;


    BDFOdeSolver();

};

}
