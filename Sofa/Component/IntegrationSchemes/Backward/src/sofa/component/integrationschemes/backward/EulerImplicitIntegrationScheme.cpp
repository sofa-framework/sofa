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
#include <sofa/component/integrationschemes/backward/EulerImplicitIntegrationScheme.h>
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

EulerImplicitIntegrationScheme::EulerImplicitIntegrationScheme()
    : d_trapezoidalScheme( initData(&d_trapezoidalScheme,false,"trapezoidalScheme","Boolean to use the trapezoidal scheme instead of the implicit Euler scheme and get second order accuracy in time (false by default)") )
    {
}



SReal EulerImplicitIntegrationScheme::getPositionUpdateDerivedFromVelocity() const
{
    return m_dt;
}

SReal EulerImplicitIntegrationScheme::getInverseVelocityUpdateDerivedFromVelocity() const
{
    return 1/m_dt;
}

//Compute the position update from current value of velocity : dX = x_t - g_x(v_i)
void EulerImplicitIntegrationScheme::computePositionUpdateFromVelocity(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity)
{
    //TODO this is not in accordance to its use in VelocityIntegrationScheme where it is ecxpected to compute X and not dX being g_x(v) - x_t
    sofa::core::behavior::MultiVecDeriv res(&vop, result );
    res.eq(velocity, m_dt);
}

//Compute the acceleration from current value of velocity. This is the implementation of the inverse integration scheme for the velocity
void EulerImplicitIntegrationScheme::computeAccelerationFromVelocity(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity)
{
    sofa::core::behavior::MultiVecDeriv res(&vop, result );
    res.eq(velocity, 1/m_dt);
    res.peq(m_v0[0], -1/m_dt);
}


void registerEulerImplicitIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator using implicit backward Euler scheme.")
        .add< EulerImplicitIntegrationScheme >());
}

} // namespace sofa::component::odesolver::backward
