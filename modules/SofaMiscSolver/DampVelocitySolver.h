/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H
#define SOFA_COMPONENT_ODESOLVER_DampVelocitySolver_H

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/SofaMisc.h>

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

    void solve (const core::ExecParams* params /* PARAMS FIRST */, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);
    Data<double> rate;
    Data<double> threshold;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    virtual double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
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
    virtual double getSolutionIntegrationFactor(int /*outputDerivative*/) const
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
