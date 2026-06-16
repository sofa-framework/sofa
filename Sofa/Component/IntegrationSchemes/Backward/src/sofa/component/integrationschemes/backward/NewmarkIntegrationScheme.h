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
#include <sofa/component/integrationschemes/backward/config.h>
#include <sofa/core/behavior/BaseIntegrationScheme.h>
#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include "sofa/simulation/integrationschemes/AccelerationBasedImplicitIntegrationScheme.h"

namespace sofa::simulation::common
{
class VectorOperations;
}
namespace sofa::component::integrationschemes::backward
{


class SOFA_COMPONENT_INTEGRATIONSCHEMES_BACKWARD_API NewmarkIntegrationScheme :
    public sofa::simulation::integrationschemes::AccelerationBasedImplicitIntegrationScheme
{
public:
    SOFA_CLASS(NewmarkIntegrationScheme, sofa::simulation::integrationschemes::AccelerationBasedImplicitIntegrationScheme);

    core::objectmodel::Data<SReal> d_beta;
    core::objectmodel::Data<SReal> d_gamma;

protected:
    NewmarkIntegrationScheme();

    SReal getPositionUpdateDerivedFromAcceleration() const override;
    SReal getPositionUpdateDerivedFromVelocity() const override;
    SReal getVelocityUpdateDerivedFromAcceleration() const override;

    //Compute the error made on the position integration equation : x_{t+h} - g_x(v,a), with v and a the current estimates of velocity and acceleration
    virtual void computeCurrentPositionIntegrationError(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity, const sofa::core::MultiVecDerivId& acceleration);
    //Compute the error made on the position integration equation : v_{t+h} - g_v(a), with a the current estimate of acceleration
    virtual void computeCurrentVelocityIntegrationError(sofa::simulation::common::VectorOperations & vop, const sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& acceleration) ;

    virtual Size getIntegrationSchemeTimeOrder() const override
    {
        return 1;
    }
};

} // namespace sofa::component::integrationschemes::backward
