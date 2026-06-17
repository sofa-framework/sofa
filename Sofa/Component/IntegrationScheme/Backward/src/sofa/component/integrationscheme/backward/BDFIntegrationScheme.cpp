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
#include <sofa/component/integrationscheme/backward/BDFIntegrationScheme.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <iomanip>


namespace sofa::component::integrationscheme::backward
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

void BDFIntegrationScheme::computeFactors()

{
    assert(m_samples.size() > 1);
    const auto order =m_samples.size() - 1;
    assert(order >= 1);

    m_aFactors.resize(order+1);
    m_bFactors.resize(order+1);

    /**
     * Computation of the derivative of the Lagrange inteperpolation polynomials
     */
    for (std::size_t j = 0; j < order+1; ++j)
    {
        auto& a_j = m_aFactors[j];

        m_bFactors[j] = (j == order) ? 1.0 : 0.0;

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
                        product *= (m_samples[order] - m_samples[m]) / (m_samples[j] - m_samples[m]);
                    }
                }
                a_j += product / (m_samples[j] - m_samples[i]);
            }
        }
    }
    for (SReal& j : m_aFactors)
    {
        j *= m_dt;
    }
}


void registerBDFIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator using implicit backward Euler scheme.")
        .add< BDFIntegrationScheme >());
}

} // namespace sofa::component::odesolver::backward
