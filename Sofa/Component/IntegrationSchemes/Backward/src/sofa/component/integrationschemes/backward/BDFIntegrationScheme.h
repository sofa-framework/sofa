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
#include <sofa/core/behavior/IntegrationScheme.h>
#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/integrationschemes/VelocityBasedIntegrationScheme.h>

#include "sofa/simulation/integrationschemes/AccelerationBasedIntegrationScheme.h"

namespace sofa::simulation::common
{
class VectorOperations;
}
namespace sofa::component::integrationschemes::backward
{


class SOFA_COMPONENT_INTEGRATIONSCHEMES_BACKWARD_API BDFIntegrationScheme :
    public sofa::simulation::integrationschemes::VelocityBasedIntegrationScheme
{
public:
    SOFA_CLASS(BDFIntegrationScheme, sofa::simulation::integrationschemes::VelocityBasedIntegrationScheme);
    core::objectmodel::Data<sofa::Size> d_order;

    BDFIntegrationScheme();

    virtual void init() override;
    virtual void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    virtual SReal getPositionUpdateDerivedFromVelocity() const override;
    virtual SReal getInverseVelocityUpdateDerivedFromVelocity() const override;

    //Compute the position update from current value of velocity : dX = g_x(v_i) - x_t
    virtual void computeCurrentPositionIntegrationError(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result,  const sofa::core::MultiVecCoordId& position,const sofa::core::MultiVecDerivId& velocity) override;
    //Compute the acceleration from current value of velocity. This is the implementation of the inverse integration scheme for the velocity
    virtual void computeAccelerationFromVelocity(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity) override;

protected:
    virtual sofa::Size getIntegrationSchemeOrder()
    {
        return d_order.getValue();
    }

    void computeAFactors();

    std::vector<SReal> m_aFactors;
    std::deque<SReal> m_samples;

};

} // namespace sofa::component::integrationschemes::backward
