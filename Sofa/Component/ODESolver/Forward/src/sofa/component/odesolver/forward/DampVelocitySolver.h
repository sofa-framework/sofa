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

#include <sofa/component/odesolver/forward/config.h>

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa::component::odesolver::forward
{

/** Velocity damping and thresholding.
This is not an ODE solver, but it can be used as a post-process after a real ODE solver.
*/
class SOFA_COMPONENT_ODESOLVER_FORWARD_API DampVelocitySolver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(DampVelocitySolver, sofa::core::behavior::OdeSolver);

    void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;
    Data<SReal> rate; ///< Factor used to reduce the velocities. Typically between 0 and 1.
    Data<SReal> threshold; ///< Threshold under which the velocities are canceled.

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    SReal getIntegrationFactor(int inputDerivative, int outputDerivative) const override
    {
        const SReal dt = getContext()->getDt();
        const SReal matrix[3][3] =
        {
            { 1, 0, },
            { 0, std::exp(-rate.getValue()*dt), 0},
            { 0, 0, 0}
        };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    SReal getSolutionIntegrationFactor(int /*outputDerivative*/) const override
    {
        return 0;
    }

protected:
    DampVelocitySolver();
};

} // namespace sofa::component::odesolver::forward
