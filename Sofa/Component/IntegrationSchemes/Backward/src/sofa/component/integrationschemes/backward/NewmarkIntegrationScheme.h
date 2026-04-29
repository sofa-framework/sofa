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


class SOFA_COMPONENT_INTEGRATIONSCHEMES_BACKWARD_API NewmarkIntegrationScheme :
    public sofa::simulation::integrationschemes::AccelerationBasedIntegrationScheme
{
public:
    SOFA_CLASS(NewmarkIntegrationScheme, sofa::simulation::integrationschemes::AccelerationBasedIntegrationScheme);

    core::objectmodel::Data<SReal> d_beta;
    core::objectmodel::Data<SReal> d_gamma;

protected:
    NewmarkIntegrationScheme();

    SReal getPositionUpdateDerivedFromAcceleration() const override;
    SReal getPositionUpdateDerivedFromVelocity() const override;
    SReal getVelocityUpdateDerivedFromAcceleration() const override;

    // void computeCurrentAccelerationFromVelocity(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity) override ;
    void computePositionUpdateFromVelocityAndAcceleration(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity, const sofa::core::MultiVecDerivId& acceleration) override;
    void computeVelocityUpdateFromAcceleration(sofa::simulation::common::VectorOperations & vop, const sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& acceleration) override;

    virtual Size getIntegrationSchemeOrder() override
    {
        return 1;
    }
};

} // namespace sofa::component::integrationschemes::backward
