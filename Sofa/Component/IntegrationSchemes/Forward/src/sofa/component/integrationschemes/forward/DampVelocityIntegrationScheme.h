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

#include <sofa/component/integrationschemes/forward/config.h>

#include <sofa/simulation/integrationschemes/ExplicitIntegrationScheme.h>

namespace sofa::component::integrationschemes::forward
{

/** Velocity damping and thresholding.
This is not an ODE IntegrationScheme, but it can be used as a post-process after a real ODE IntegrationScheme.
*/
class SOFA_COMPONENT_INTEGRATIONSCHEMES_FORWARD_API DampVelocityIntegrationScheme : public simulation::integrationschemes::ExplicitIntegrationScheme
{
public:
    SOFA_CLASS(DampVelocityIntegrationScheme, simulation::integrationschemes::ExplicitIntegrationScheme);

    Data<SReal> d_rate; ///< Factor used to reduce the velocities. Typically between 0 and 1.
    Data<SReal> d_threshold; ///< Threshold under which the velocities are canceled.

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual void doIntegrate(const core::ExecParams* params, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    virtual SReal getVelocityIntegrationFactor() const override
    {
        return 1.0;
    }

    virtual SReal getPositionIntegrationFactor() const override
    {
        return m_dt;
    }
protected:
    DampVelocityIntegrationScheme();
};

} // namespace sofa::component::odeIntegrationScheme::forward
