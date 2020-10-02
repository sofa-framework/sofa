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

#include "config.h"

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** The simplest time integration.
 Two variants are available, depending on the value of field "symplectic".
 If true (the default), the symplectic variant of Euler's method is applied:
 If false, the basic Euler's method is applied (less robust)
 */
class SOFA_EXPLICIT_ODE_SOLVER_API EulerExplicitSolver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(EulerExplicitSolver, sofa::core::behavior::OdeSolver);
protected:
    EulerExplicitSolver();
public:
    void solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    Data<bool> d_symplectic; ///< If true, the velocities are updated before the positions and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).
    Data<bool> d_optimizedForDiagonalMatrix; ///< If M matrix is sparse (MeshMatrixMass), must be set to false (function addMDx() will compute the mass). Else, if true, solution to the system Ax=b can be directly found by computing x = f/m. The function accFromF() in the mass API will be used.
    Data<bool> d_threadSafeVisitor;

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    double getIntegrationFactor(int inputDerivative, int outputDerivative) const override ;

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    double getSolutionIntegrationFactor(int outputDerivative) const override ;
    void init() override ;


protected:
    /// the solution vector is stored for warm-start
    core::behavior::MultiVecDeriv x;
};

} // namespace odesolver

} // namespace component

} // namespace sofa
