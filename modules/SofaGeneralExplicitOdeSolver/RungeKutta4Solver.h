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
#ifndef SOFA_COMPONENT_ODESOLVER_RUNGEKUTTA4SOLVER_H
#define SOFA_COMPONENT_ODESOLVER_RUNGEKUTTA4SOLVER_H
#include "config.h"

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** A popular time integration method, much more precise than the EulerSolver */
class SOFA_GENERAL_EXPLICIT_ODE_SOLVER_API RungeKutta4Solver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(RungeKutta4Solver, sofa::core::behavior::OdeSolver);

    void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    /// @todo use real factors depending on the current RK4 step
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const override
    {
        const SReal dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, dt/2, 0},
            { 0, 1, dt/2},
            { 0, 0, 0}
        };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    /// @todo use real factors depending on the current RK4 step
    virtual double getSolutionIntegrationFactor(int outputDerivative) const override
    {
        const SReal dt = getContext()->getDt();
        double vect[3] = { 0.0, dt/2, 1};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }

};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
