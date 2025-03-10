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
#include <sofa/component/odesolver/backward/BDFOdeSolver.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::odesolver::backward
{

void registerBDFOdeSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        core::ObjectRegistrationData("Velocity-based ODE solver using Backward Differentiation Formula (BDF), at any order, supporting variable time step size.")
            .add<BDFOdeSolver>());
}

void BDFOdeSolver::recomputeCoefficients(std::size_t order, SReal dt)
{
    assert(order > 0);

    m_a_coef.resize(order+1);
    m_b_coef.resize(order+1);

    /**
     * Computation of the derivative of the Lagrange inteperpolation polynomials
     */
    for (std::size_t j = 0; j < m_a_coef.size(); ++j)
    {
        auto& a_j = m_a_coef[j];

        a_j = 0;
        for (std::size_t i = 0; i < order+1; ++i)
        {
            if (i != j)
            {
                SReal product = 1_sreal;
                for (std::size_t m = 0; m < order + 1; ++m)
                {
                    if (m != i && m != j)
                    {
                        product *= (m_timeList[order] - m_timeList[m]) / (m_timeList[j] - m_timeList[m]);
                    }
                }
                a_j += product / (m_timeList[j] - m_timeList[i]);
            }
        }
    }

    for (SReal& j : m_a_coef)
    {
        j *= dt;
    }

    for (std::size_t j = 0; j < m_a_coef.size() - 1; ++j)
    {
        m_a_coef[j] /= m_a_coef[order];
    }
    m_b_coef[order] = 1 / m_a_coef[order];
    m_a_coef[order] = 1;

    for (std::size_t j = 0; j < m_b_coef.size() - 1; ++j)
    {
        m_b_coef[j] = 0;
    }
}

}  // namespace sofa::component::odesolver::backward
