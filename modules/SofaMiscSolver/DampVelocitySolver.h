/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H
#define SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H
#include "config.h"

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** Velocity damping and thresholding.
This is not an ODE solver, but it can be used as a post-process after a real ODE solver.
*/
class SOFA_MISC_SOLVER_API DampVelocitySolver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(DampVelocitySolver, sofa::core::behavior::OdeSolver);

    void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;
    Data<double> rate;
    Data<double> threshold;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const override
    {
        const SReal dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, 0, 0},
            { 0, exp(-rate.getValue()*dt), 0},
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
    virtual double getSolutionIntegrationFactor(int /*outputDerivative*/) const override
    {
        return 0;
    }

protected:
    DampVelocitySolver();
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
