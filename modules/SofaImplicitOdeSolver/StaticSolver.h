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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_ODESOLVER_STATICSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_STATICSOLVER_H

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/SofaCommon.h>


namespace sofa
{

namespace component
{

namespace odesolver
{

/** Finds the static equilibrium of a system. Can diverge when there are an infinity of solutions. */
class SOFA_IMPLICIT_ODE_SOLVER_API StaticSolver : public sofa::core::behavior::OdeSolver
{

public:
    SOFA_CLASS(StaticSolver, sofa::core::behavior::OdeSolver);
protected:
    StaticSolver();
public:
    void solve (const core::ExecParams* params /* PARAMS FIRST */, double dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    double getIntegrationFactor(int inputDerivative, int outputDerivative) const
    {
        double matrix[3][3] =
        {
            { 1, 0, 0},
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
        double vect[3] = { 1, 0, 0};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }

    Data<double> massCoef;
    Data<double> dampingCoef;
    Data<double> stiffnessCoef;
    Data<bool> applyIncrementFactor; ///< multiply the solution by dt. Default: false
};

} // namespace odesolver

} // namespace component

} // namespace sofa

#endif
