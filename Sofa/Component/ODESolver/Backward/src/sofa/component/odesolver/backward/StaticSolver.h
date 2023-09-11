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

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/MultiVec.h>

namespace sofa::component::odesolver::backward
{

using sofa::core::objectmodel::Data;

/**
 * Implementation of a static ODE solver compatible with non-linear materials.
 *
 * We are trying to solve to following
 * \f{eqnarray*}{
 *     \vec{R}(\vec{x}) - \vec{P} = 0
 * \f}
 *
 * Where \f$\vec{R}\f$ is the (possibly non-linear) internal elastic force residual and \f$\vec{P}\f$ is the external
 * force vector (for example, gravitation force or surface traction).
 *
 * Following the <a href="https://en.wikipedia.org/wiki/Newton's_method#Nonlinear_systems_of_equations">Newton-Raphson method</a>,
 * we pose
 *
 * \f{align*}{
 *     \vec{F}(\vec{x}_{n+1}) &= \vec{R}(\vec{x}_{n+1}) - \vec{P}_n \\
 *     \mat{J} = \frac{\partial \vec{F}}{\partial \vec{x}_{n+1}} \bigg\rvert_{\vec{x}_{n+1}^i} &= \mat{K}(\vec{x}_{n+1})
 * \f}
 *
 * where \f$\vec{x}_{n+1}\f$ is the unknown position vector at the \f$n\f$th time step. We then iteratively solve
 *
 * \f{align*}{
 *     \mat{K}(\vec{x}_{n+1}^i) \left [ \Delta \vec{x}_{n+1}^{i+1} \right ] &= - \vec{F}(\vec{x}_{n+1}^i) \\
 *     \vec{x}_{n+1}^{i+1} &= \vec{x}_{n+1}^{i} + \Delta \vec{x}_{n+1}^{i+1}
 * \f}
 */
class SOFA_COMPONENT_ODESOLVER_BACKWARD_API StaticSolver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(StaticSolver, sofa::core::behavior::OdeSolver);
    StaticSolver();

public:
    void solve (const sofa::core::ExecParams* params /* PARAMS FIRST */, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    /** The list of squared residual norms (r.dot(r) = ||r||^2) of every newton iterations of the last solve call. */
    auto squared_residual_norms() const -> const std::vector<SReal> & { return p_squared_residual_norms; }

    /** The list of squared correction increment norms (dx.dot(dx) = ||dx||^2) of every newton iterations of the last solve call. */
    auto squared_increment_norms() const -> const std::vector<SReal> & { return p_squared_increment_norms; }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt.
    SReal getVelocityIntegrationFactor() const override
    {
        return 1.0; // getContext()->getDt();
    }

    /// Given a displacement as computed by the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    /// For Euler methods, it is typically dt².
    SReal getPositionIntegrationFactor() const override
    {
        return getPositionIntegrationFactor(getContext()->getDt());
    }

    virtual SReal getPositionIntegrationFactor(SReal dt ) const
    {
        return dt;
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
    SReal getIntegrationFactor(int inputDerivative, int outputDerivative) const override
    {
        return getIntegrationFactor(inputDerivative, outputDerivative, getContext()->getDt());
    }

    SReal getIntegrationFactor(int inputDerivative, int outputDerivative, SReal dt) const
    {
        const SReal matrix[3][3] =
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
    SReal getSolutionIntegrationFactor(int outputDerivative) const override
    {
        return getSolutionIntegrationFactor(outputDerivative, getContext()->getDt());
    }

    SReal getSolutionIntegrationFactor(int outputDerivative, SReal dt) const
    {
        const SReal vect[3] = { dt, 1, 1/dt};
        if (outputDerivative >= 3)
            return 0;
        else
            return vect[outputDerivative];
    }

protected:

    Data<unsigned> d_newton_iterations; ///< Number of newton iterations between each load increments (normally, one load increment per simulation time-step.
    Data<SReal> d_absolute_correction_tolerance_threshold; ///< Convergence criterion: The newton iterations will stop when the norm |du| is smaller than this threshold.
    Data<SReal> d_relative_correction_tolerance_threshold; ///< Convergence criterion: The newton iterations will stop when the ratio |du| / |U| is smaller than this threshold.
    Data<SReal> d_absolute_residual_tolerance_threshold; ///< Convergence criterion: The newton iterations will stop when the norm of the residual |R| is smaller than this threshold. Use a negative value to disable this criterion.
    Data<SReal> d_relative_residual_tolerance_threshold; ///< Convergence criterion: The newton iterations will stop when the ratio |R|/|R0| is smaller than this threshold. Use a negative value to disable this criterion.
    Data<bool> d_should_diverge_when_residual_is_growing; ///< Divergence criterion: The newton iterations will stop when the residual is greater than the one from the previous iteration.

private:
    /// Sum of displacement increments since the beginning of the time step
    sofa::core::behavior::MultiVecDeriv U;

    /// List of squared residual norms (r.dot(R) = ||r||^2) of every newton iterations of the last solve call.
    std::vector<SReal> p_squared_residual_norms;

    /// List of squared correction increment norms (dx.dot(dx) = ||dx||^2) of every newton iterations of the last solve call.
    std::vector<SReal> p_squared_increment_norms;
};

} // namespace sofa::component::odesolver::backward
