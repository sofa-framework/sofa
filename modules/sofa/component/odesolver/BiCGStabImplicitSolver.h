/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
// Author: Jeremie Allard, Sim Group @ CIMIT, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_ODESOLVER_BICGSTABIMPLICITSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_BICGSTABIMPLICITSOLVER_H

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/component/odesolver/OdeSolverImpl.h>

namespace sofa
{

namespace component
{

namespace odesolver
{
/** Implicit integration solver able to handle degenerate equation systems.
*/
class SOFA_COMPONENT_ODESOLVER_API BiCGStabImplicitSolver : public sofa::component::odesolver::OdeSolverImpl
{
public:
    SOFA_CLASS(BiCGStabImplicitSolver, sofa::component::odesolver::OdeSolverImpl);
    typedef core::behavior::OdeSolver Inherited;

    BiCGStabImplicitSolver();
    void solve (double dt);
//     BiCGStabImplicitSolver* setMaxIter( int maxiter );

    Data<unsigned int> maxCGIter;
    Data<double> smallDenominatorThreshold;
    Data<double> tolerance;
    Data<double> rayleighStiffness;

    bool getDebug()
    {
        return false;
    }

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    ///
    /// This method is used to compute the compliance for contact corrections.
    /// For example, a backward-Euler dynamic implicit integrator would use:
    /// Input:      x_t  v_t  a_{t+dt}
    /// x_{t+dt}     1    dt  dt^2
    /// v_{t+dt}     0    1   dt
    ///
    /// If the linear system is expressed on s = a_{t+dt} dt, then the final factors are:
    /// Input:      x_t   v_t    a_t  s
    /// x_{t+dt}     1    dt     0    dt
    /// v_{t+dt}     0    1      0    1
    /// a_{t+dt}     0    0      0    1/dt
    /// The last column is returned by the getSolutionIntegrationFactor method.
    double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double matrix[3][3] =
        {
            { 1, dt, 0},
            { 0, 1, 0},
            { 0, 0, 0}
        };
        if (inputDerivative >= 3 || outputDerivative >= 3)
            return 0;
        else
            return matrix[outputDerivative][inputDerivative];
    }

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    double getSolutionIntegrationFactor(int outputDerivative) const
    {
        const double dt = getContext()->getDt();
        double vect[3] = { dt, 1, 1/dt};
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


