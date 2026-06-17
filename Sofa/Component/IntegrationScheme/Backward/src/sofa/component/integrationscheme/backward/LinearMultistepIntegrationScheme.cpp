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
#include <sofa/component/integrationscheme/backward/LinearMultistepIntegrationScheme.h>
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

LinearMultistepIntegrationScheme::LinearMultistepIntegrationScheme()
: d_order(initData(&d_order,Size(2),"order","Order of the Backward Differential Formula.") )
{  }

void LinearMultistepIntegrationScheme::init()
{
    Inherit1::init();
    if (d_order.getValue() == 0)
    {
        msg_error()<<"Order cannot be null";
        d_componentState.setValue(core::objectmodel::ComponentState::Invalid);
    }
}

void LinearMultistepIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt,
                                                  sofa::core::MultiVecCoordId xResult,
                                                  sofa::core::MultiVecDerivId vResult)
{
    Inherit1::doSetupIntegrationStep(params, dt, xResult, vResult);

    if (m_samples.empty())
    {
        for (unsigned i = 0; i < d_order.getValue() + 1; i++)
            m_samples.push_front(- i * m_dt);
    }

    m_samples.pop_front();
    m_samples.push_back(m_samples.back() + m_dt);

    computeFactors ();
}

SReal LinearMultistepIntegrationScheme::getPositionUpdateDerivedFromVelocity() const
{
    return m_dt *  m_bFactors[d_order.getValue()] / m_aFactors[d_order.getValue()];
}

SReal LinearMultistepIntegrationScheme::getInverseVelocityUpdateDerivedFromVelocity() const
{
    return 1.0 / m_dt ;
}

//Compute the error made on the position integration equation : x_{t+h} - g_x(v), with v the current estimate of velocity
void LinearMultistepIntegrationScheme::computeCurrentPositionIntegrationError(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result,  const sofa::core::MultiVecCoordId& position, const sofa::core::MultiVecDerivId& velocity)
{

    sofa::core::behavior::MultiVecDeriv res(&vop, result );
    res.eq(velocity, - m_dt * m_bFactors[d_order.getValue()]/ m_aFactors[d_order.getValue()]);
    for (unsigned i = 0; i < d_order.getValue(); i++)
    {
        if ( fabs(m_bFactors[i]) > std::numeric_limits<SReal>::epsilon())
            res.peq(m_v0[i], - m_dt * m_bFactors[i]/m_aFactors[d_order.getValue()]);
    }
    for (unsigned i = 0; i < d_order.getValue(); i++)
    {
        //TODO How does that work in practice ? Deriv += f * Coord
        res.peq(m_x0[i], m_aFactors[i]/m_aFactors[d_order.getValue()]);
    }
    res.peq(position);

}
//Compute the acceleration from current value of velocity. This is the implementation of the inverse integration scheme for the velocity
void LinearMultistepIntegrationScheme::computeAccelerationFromVelocity(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity)
{

    sofa::core::behavior::MultiVecDeriv res(&vop, result );
    res.eq(velocity, 1/m_dt);
    res.peq(m_v0[d_order.getValue() - 1], -1/m_dt);

}



} // namespace sofa::component::odesolver::backward
