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
#include <sofa/component/integrationschemes/backward/NewmarkIntegrationScheme.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <iomanip>


namespace sofa::component::integrationschemes::backward
{

using core::VecId;
using namespace sofa::defaulttype;
using namespace core::behavior;

NewmarkIntegrationScheme::NewmarkIntegrationScheme()
: d_beta(initData(&d_beta,0.25,"beta","Factor controlling the 'implicitness' of the position computation with respect to the acceleration. 0.0 means explicit central difference, 1.0/0.6 means linear accelerations scheme") )
, d_gamma(initData(&d_gamma,0.5,"gamma","Factor controlling the 'implicitness' of the velocity computation with respect to the acceleration. To insure unconditional stability, gamma must belong to [2*beta, 1/2]. ") )
{  }

SReal NewmarkIntegrationScheme::getPositionUpdateDerivedFromAcceleration() const
{
    return m_dt * m_dt * d_beta.getValue();
}

SReal NewmarkIntegrationScheme::getPositionUpdateDerivedFromVelocity() const
{
    return 0.0;
}

SReal NewmarkIntegrationScheme::getVelocityUpdateDerivedFromAcceleration() const
{
    return m_dt * d_gamma.getValue();
}


void NewmarkIntegrationScheme::computeCurrentPositionIntegrationError(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity, const sofa::core::MultiVecDerivId& acceleration)
{
    sofa::core::behavior::MultiVecDeriv res(&vop, result );
    res.eq(m_xResult,m_x0[0], -1);
    res.peq(m_v0[0], -m_dt);
    res.peq(acceleration, -m_dt * m_dt * d_beta.getValue() );
    res.peq(m_a0[0], -m_dt * m_dt * (0.5  -  d_beta.getValue()) );
}

void NewmarkIntegrationScheme::computeCurrentVelocityIntegrationError(sofa::simulation::common::VectorOperations & vop, const sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& acceleration)
{
    sofa::core::behavior::MultiVecDeriv res(&vop, result );
    res.eq( m_vResult, m_v0[0], -1);
    res.peq(acceleration, -m_dt * d_gamma.getValue() );
    res.peq(m_a0[0], -m_dt * (1.0  -  d_gamma.getValue()) );
}


void registerNewmarkIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator using implicit backward Euler scheme.")
        .add< NewmarkIntegrationScheme >());
}

} // namespace sofa::component::odesolver::backward
