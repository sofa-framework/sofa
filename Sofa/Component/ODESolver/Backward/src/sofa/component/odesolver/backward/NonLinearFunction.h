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
#include <sofa/component/odesolver/backward/config.h>

#include <memory>
#include <stdexcept>

namespace sofa::component::odesolver::backward::newton_raphson
{

/**
 * Base class of a representation of a nonlinear function in the context of the resolution of a
 * nonlinear (system of) equation(s) with the Newton-Raphson method.
 *
 * In the objective to be very generic, the computations required by the Newton-Raphson are defined
 * inside @BaseNonLinearFunction. For example, a Newton iteration requires to solve a linear
 * equation that can be solved using a linear solver in the case of a system of equations, or by a
 * scalar inversion in the case of a scalar equation. The only required interaction with the outside
 * is the squared norm of the last evaluation.
 *
 * If r is the nonlinear function, a Newton iteration leads to the following linear equation:
 * J_r (x^{i+1} - x^i) = -r(x^i)
 * where J_r is the Jacobian of r and x^i is the current guess of r at iteration i.
 */
class SOFA_COMPONENT_ODESOLVER_BACKWARD_API BaseNonLinearFunction
{
public:
    virtual ~BaseNonLinearFunction() = default;

    virtual void startNewtonIteration() {}
    virtual void endNewtonIteration() {}

    /**
     * Evaluation of the function where the input is the current guess. If the function is called
     * for the first time, then it is called on the initial guess.
     * The evaluation is computed internally. It is not necessary to share this evaluation with the
     * outside.
     */
    virtual void evaluateCurrentGuess() = 0;

    /**
     * Returns the squared norm of the last evaluation of the function
     */
    virtual SReal squaredNormLastEvaluation() = 0;

    /**
     * Compute the gradient internally. It is not necessary to share this gradient with the outside
     */
    virtual void computeGradientFromCurrentGuess() = 0;

    /**
     * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
     * It is solved internally. It is not necessary to share the result with the outside
     */
    virtual void solveLinearEquation() = 0;

    /**
     * Once (x^{i+1}-x^i) has been computed, the result is used internally to update the current
     * guess. It computes x^{i+1} += alpha * dx, where dx is the result of the linear system. It is
     * not necessary to share the result with the Newton-Raphson method.
     */
    virtual void updateGuessFromLinearSolution(SReal alpha) = 0;

    /**
     * Compute ||x^{i+1}-x^i||^2
     */
    virtual SReal squaredNormDx() = 0;

    /**
     * Compute ||x^{i+1}||^2
     */
    virtual SReal squaredLastEvaluation() = 0;
};

}
