/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_ODESOLVER_NEWMARKIMPLICITSOLVER_H
#define SOFA_COMPONENT_ODESOLVER_NEWMARKIMPLICITSOLVER_H
#include "config.h"

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa
{

namespace component
{

namespace odesolver
{

/** Implicit time integrator using Newmark scheme.
 *
 * This integration scheme is based on the following equations:
 *
 *   $x_{t+h} = x_t + h v_t + h^2/2 ( (1-2\beta) a_t + 2\beta a_{t+h} )$
 *   $v_{t+h} = v_t + h ( (1-\gamma) a_t + \gamma a_{t+h} )$
 *
 * Applied to a mechanical system where $ M a_t + (r_M M + r_K K) v_t + K x_t = f_ext$, we need to solve the following system:
 *
 *   $ M a_{t+h} + (r_M M + r_K K) v_{t+h} + K x_{t+h} = f_ext $
 *   $ M a_{t+h} + (r_M M + r_K K) ( v_t + h ( (1-\gamma) a_t + \gamma a_{t+h} ) ) + K ( x_t + h v_t + h^2/2 ( (1-2\beta) a_t + 2\beta a_{t+h} ) ) = f_ext $
 *   $ ( M + h \gamma (r_M M + r_K K) + h^2 \beta K ) a_{t+h} = f_ext - (r_M M + r_K K) ( v_t + h (1-\gamma) a_t ) - K ( x_t + h v_t + h^2/2 (1-2\beta) a_t ) $
 *   $ ( (1 + h \gamma r_M) M + (h^2 \beta + h \gamma r_K) K ) a_{t+h} = f_ext - (r_M M + r_K K) v_t - K x_t - (r_M M + r_K K) ( h (1-\gamma) a_t ) - K ( h v_t + h^2/2 (1-2\beta) a_t ) $
 *   $ ( (1 + h \gamma r_M) M + (h^2 \beta + h \gamma r_K) K ) a_{t+h} = a_t - (r_M M + r_K K) ( h (1-\gamma) a_t ) - K ( h v_t + h^2/2 (1-2\beta) a_t ) $
 *
 * The current implementation first computes $a_t$ directly (as in the explicit solvers), then solves the previous system to compute $a_{t+dt}$, and finally computes the new position and velocity.
 *
*/
class SOFA_MISC_SOLVER_API NewmarkImplicitSolver : public sofa::core::behavior::OdeSolver
{
protected:
    unsigned int cpt;
    sofa::core::MultiVecDerivId pID;

    NewmarkImplicitSolver();

public:
    SOFA_CLASS(NewmarkImplicitSolver, sofa::core::behavior::OdeSolver);
    Data<double> f_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness
    Data<double> f_rayleighMass; ///< Rayleigh damping coefficient related to mass
    Data<double> f_velocityDamping; ///< Velocity decay coefficient (no decay if null)
    Data<bool> f_verbose; ///< Dump system state at each iteration

    Data<double> f_gamma; ///< Newmark scheme gamma coefficient
    Data<double> f_beta; ///< Newmark scheme beta coefficient

    void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
    virtual double getVelocityIntegrationFactor() const override
    {
        return 1.0; // getContext()->getDt();
    }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the position
    virtual double getPositionIntegrationFactor() const override
    {
        return getContext()->getDt(); //*getContext()->getDt());
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
    double getIntegrationFactor(int inputDerivative, int outputDerivative) const override
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
    double getSolutionIntegrationFactor(int outputDerivative) const override
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
