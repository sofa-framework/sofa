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
#include <sofa/component/odesolver/forward/config.h>

#include <sofa/core/behavior/OdeSolver.h>
#include <sofa/core/behavior/MultiVec.h>

namespace sofa::simulation::common
{
class MechanicalOperations;
class VectorOperations;
}

namespace sofa::core::behavior
{
template<class T>
class MultiMatrix;
}

namespace sofa::component::odesolver::forward
{

/**
 * The simplest time integration.
 * Two variants of the Euler solver are available in this component:
 * - forward Euler method, also called explicit Euler method
 * - semi-implicit Euler method, also called semi-explicit Euler method or symplectic Euler
 *
 * In both variants, acceleration is first computed. The system to compute the acceleration
 * is M * a = f, where M is the mass matrix and f can be a force.
 * In case of a diagonal mass matrix, M is trivially invertible and the acceleration
 * can be computed without a linear solver.
 *
 * f is accumulated by force fields through the addForce function. Mappings can
 * also contribute by projecting forces of mapped objects.
 * f is computed based on the current state (current velocity and position).
 *
 * Explicit Euler method:
 * The option "symplectic" must be set to false to use this variant.
 * The explicit Euler method produces an approximate discrete solution by iterating
 * x_{n+1} = x_n + v_n * dt
 * v_{n+1} = v_n + a * dt
 *
 * Semi-implicit Euler method:
 * The option "symplectic" must be set to true to use this variant.
 * The semi-implicit Euler method produces an approximate discrete solution by iterating
 * v_{n+1} = v_n + a * dt
 * x_{n+1} = x_n + v_{n+1} * dt
 *
 * The semi-implicit Euler method is more robust than the standard Euler method.
 */
class SOFA_COMPONENT_ODESOLVER_FORWARD_API EulerExplicitSolver : public sofa::core::behavior::OdeSolver
{
public:
    SOFA_CLASS(EulerExplicitSolver, sofa::core::behavior::OdeSolver);

protected:
    EulerExplicitSolver();

public:
    void solve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    Data<bool> d_symplectic; ///< If true, the velocities are updated before the positions and the method is symplectic (more robust). If false, the positions are updated before the velocities (standard Euler, less robust).
    Data<bool> d_threadSafeVisitor; ///< If true, do not use realloc and free visitors in fwdInteractionForceField.

    /// Given an input derivative order (0 for position, 1 for velocity, 2 for acceleration),
    /// how much will it affect the output derivative of the given order.
    SReal getIntegrationFactor(int inputDerivative, int outputDerivative) const override ;

    /// Given a solution of the linear system,
    /// how much will it affect the output derivative of the given order.
    ///
    SReal getSolutionIntegrationFactor(int outputDerivative) const override ;

    void init() override ;

    void parse(sofa::core::objectmodel::BaseObjectDescription* arg) override;

protected:

    /// Update state variable (new position and velocity) based on the computed acceleration
    /// The update takes constraints into account
    void updateState(sofa::simulation::common::VectorOperations* vop,
                     sofa::simulation::common::MechanicalOperations* mop,
                     sofa::core::MultiVecCoordId xResult,
                     sofa::core::MultiVecDerivId vResult,
                     const sofa::core::behavior::MultiVecDeriv& acc,
                     SReal dt) const;

    /// Gravity times time step size is added to the velocity for some masses
    /// v += g * dt
    static void addSeparateGravity(sofa::simulation::common::MechanicalOperations* mop, SReal dt, core::MultiVecDerivId v);

    /// Assemble the force vector (right-hand side of the equation)
    static void computeForce(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId f);

    /// Compute the acceleration from the force and the inverse of the mass
    /// acc = M^-1 * f
    static void computeAcceleration(sofa::simulation::common::MechanicalOperations* mop,
                                    core::MultiVecDerivId acc,
                                    core::ConstMultiVecDerivId f);

    /// Apply projective constraints, such as FixedProjectiveConstraint
    static void projectResponse(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId vecId);

    static void solveConstraints(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId acc);

    static void assembleSystemMatrix(core::behavior::MultiMatrix<simulation::common::MechanicalOperations>* matrix);

    static void solveSystem(core::behavior::MultiMatrix<simulation::common::MechanicalOperations>* matrix,
                            core::MultiVecDerivId solution, core::MultiVecDerivId rhs);
};

} // namespace sofa::component::odesolver::forward
