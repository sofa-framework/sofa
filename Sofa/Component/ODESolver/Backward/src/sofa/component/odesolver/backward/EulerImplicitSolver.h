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
#include <sofa/core/behavior/LinearSolverAccessor.h>

#include <sofa/core/behavior/OdeSolver.h>

namespace sofa::component::odesolver::backward
{

/** Semi-implicit time integrator using backward Euler scheme for first and second degree ODEs. (default: second)
 *
 *** 2nd Order ***
 *
 * This is based on [Baraff and Witkin, Large Steps in Cloth Simulation, SIGGRAPH 1998]
 * The integration scheme is based on the following equations:
 *
 *   \f$x_{t+h} = x_t + h v_{t+h}\f$
 *   \f$v_{t+h} = v_t + h a_{t+h}\f$
 *
 *   The unknown is
 *   \f$v_{t+h} - v_t = dv\f$
 *
 *   Newton's law is
 *   \f$ M dv = h f(t+h) \f$
 *   \f$ M dv = h ( f(t) + K dx     + (B - r_M M + r_K K) (v+dv) )\f$
 *   \f$ M dv = h ( f(t) + K h (v+dv) + (B - r_M M + r_K K) (v+dv) )\f$
 *
 *   \f$ M \f$ is the mass matrix.
 *   \f$ K = df/dx \f$ is the stiffness implemented (or not) by the force fields.
 *   \f$ B = df/dv \f$ is the damping implemented (or not) by the force fields.
 *   An additional, uniform Rayleigh damping  \f$- r_M M + r_K K\f$ is imposed by the solver.
 *
 * This corresponds to the following equation system:
 *
 *   \f$ ( (1+h r_M) M - h B - h(h + r_K) K ) dv = h ( f(t) + (h+r_K) K v + B v - r_M M v )\f$
 *
 * Moreover, the projective constraints filter out the forbidden motions.
 * This is equivalent with multiplying vectors with a projection matrix \f$P\f$.
 * Finally, the equation system set by this ode solver is:
 *
 *   \f$ P ( (1+h r_M) M - h B - h(h + r_K) K ) P dv = P h ( f(t) + (h + r_K) K v + B v - r_M M v )\f$
 *
 *** 1st Order ***
 *
 * This integration scheme is based on the following equation:
 *
 *   \f$x_{t+h} = x_t + h v_{t+h}\f$
 *
 * Applied to this mechanical system:
 *
 *   \f$ M v_t = f_{ext} \f$
 *
 *   \f$ M v_{t+h} = f_{ext_{t+h}} \f$
 *   \f$           = f_{ext_{t}} + h (df_{ext}/dt)_{t+h} \f$
 *   \f$           = f_{ext_{t}} + h (df_{ext}/dx)_{t+h} v_{t+h} \f$
 *   \f$           = f_{ext_{t}} - h K v_{t+h} \f$
 *
 *   \f$ ( M + h K ) v_{t+h} = f_{ext} \f$
 *
 *
 *** Trapezoidal Rule ***
 *
 * The trapezoidal scheme is based on
 *
 *   \f$v_{t+h} = h/2 ( f(t+h) + f(t) )\f$
 *
 * With this and the same techniques as for the implicit Euler scheme we receive for *** 2nd Order *** equations
 *
 *   \f$ P ( (1+h/2 r_M) M - h/2 B - h/2 (h + r_K) K ) P dv = P h/2 ( 2 f(t) + (h + r_K) K v + B v - r_M M v )\f$
 *
 * and for *** 1st Order ***
 *
 *   \f$ ( M + h/2 K ) v_{t+h} = f_{ext} \f$
 *
 */
class SOFA_COMPONENT_ODESOLVER_BACKWARD_API EulerImplicitSolver :
    public sofa::core::behavior::OdeSolver,
    public sofa::core::behavior::LinearSolverAccessor
{
public:
    SOFA_CLASS2(EulerImplicitSolver, sofa::core::behavior::OdeSolver, sofa::core::behavior::LinearSolverAccessor);

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_rayleighStiffness;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_rayleighMass;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<SReal> f_velocityDamping;

    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<bool> f_firstOrder;


    SOFA_ATTRIBUTE_DEPRECATED__RENAME_DATA_IN_ODESOLVER_BACKWARD()
    Data<bool> f_solveConstraint;



    Data<SReal> d_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0
    Data<SReal> d_velocityDamping; ///< Velocity decay coefficient (no decay if null)
    Data<bool> d_firstOrder; ///< Use backward Euler scheme for first order ode system.
    Data<bool> d_trapezoidalScheme; ///< Optional: use the trapezoidal scheme instead of the implicit Euler scheme and get second order accuracy in time
    Data<bool> d_solveConstraint; ///< Apply ConstraintSolver (requires a ConstraintSolver in the same node as this solver, disabled by by default for now)
    Data<bool> d_threadSafeVisitor; ///< If true, do not use realloc and free visitors in fwdInteractionForceField.

protected:
    EulerImplicitSolver();
public:
    void init() override;

    void cleanup() override;

    void solve (const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt.
    SReal getVelocityIntegrationFactor() const override { return 1.0; }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt².
    SReal getPositionIntegrationFactor() const override ;

    virtual SReal getPositionIntegrationFactor(SReal dt ) const { return dt; }

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
    SReal getIntegrationFactor(int inputDerivative, int outputDerivative) const override ;

    SReal getIntegrationFactor(int inputDerivative, int outputDerivative, SReal dt) const ;

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    SReal getSolutionIntegrationFactor(int outputDerivative) const override ;

    SReal getSolutionIntegrationFactor(int outputDerivative, SReal dt) const ;

protected:

    /// the solution vector is stored for warm-start
    core::behavior::MultiVecDeriv x;

};

} // namespace sofa::component::odesolver::backward
