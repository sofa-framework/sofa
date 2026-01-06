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
    factory->registerObjects(core::ObjectRegistrationData(
        "Velocity-based ODE solver using Backward Differentiation Formula (BDF), at any order, supporting variable time step size.")
        .add<BDFOdeSolver>());
}

BDFOdeSolver::BDFOdeSolver()
{
    this->addUpdateCallback("checkOrder", {&d_order}, [this](const core::DataTracker& )
    {
        auto order = sofa::helper::getWriteAccessor(d_order);
        if (order > 6)
        {
            msg_warning() << "The method with order " << order << " is not zero-stable";
        }
        else if (order == 0)
        {
            msg_warning() << "The method cannot have order 0. Setting it to 1.";
            order.wref() = 1;
        }
        return this->getComponentState();
    }, {});
}

void BDFOdeSolver::computeLinearMultiStepCoefficients(const std::deque<SReal>& samples,
                                                      sofa::type::vector<SReal>& a_coef,
                                                      sofa::type::vector<SReal>& b_coef)
{
    assert(samples.size() > 1);
    const auto order = samples.size() - 1;
    assert(order >= 1);

    a_coef.resize(order+1);
    b_coef.resize(order+1);

    /**
     * Computation of the derivative of the Lagrange inteperpolation polynomials
     */
    for (std::size_t j = 0; j < order+1; ++j)
    {
        auto& a_j = a_coef[j];

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
                        product *= (samples[order] - samples[m]) / (samples[j] - samples[m]);
                    }
                }
                a_j += product / (samples[j] - samples[i]);
            }
        }
    }

    const auto dt = samples.at(samples.size() - 1) - samples.at(samples.size() - 2);

    for (SReal& j : a_coef)
    {
        j *= dt;
    }

    assert(a_coef[order] != 0);
    for (std::size_t j = 0; j < a_coef.size() - 1; ++j)
    {
        a_coef[j] /= a_coef[order];
    }
    b_coef[order] = 1 / a_coef[order];
    a_coef[order] = 1;

    for (std::size_t j = 0; j < b_coef.size() - 1; ++j)
    {
        b_coef[j] = 0;
    }
}

void BDFOdeSolver::recomputeCoefficients(std::size_t order, SReal dt)
{
    assert(m_timeList.size() == order + 1);

    SOFA_UNUSED(order);
    SOFA_UNUSED(dt);
    
    computeLinearMultiStepCoefficients(m_timeList, m_a_coef, m_b_coef);
}

}  // namespace sofa::component::odesolver::backward
