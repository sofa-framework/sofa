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
#include <sofa/component/odesolver/forward/EulerExplicitSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalGetNonDiagonalMassesCountVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalGetNonDiagonalMassesCountVisitor;

//#define SOFA_NO_VMULTIOP

namespace sofa::component::odesolver::forward
{

using namespace sofa::defaulttype;
using namespace sofa::helper;
using namespace core::behavior;

void registerEulerExplicitSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("A simple explicit time integrator.")
        .add< EulerExplicitSolver >());
}

EulerExplicitSolver::EulerExplicitSolver()
    : d_symplectic( initData( &d_symplectic, true, "symplectic", "If true (default), the velocities are updated before the positions and the method is symplectic, more robust. If false, the positions are updated before the velocities (standard Euler, less robust).") )
    , d_threadSafeVisitor(initData(&d_threadSafeVisitor, false, "threadSafeVisitor", "If true, do not use realloc and free visitors in fwdInteractionForceField."))
    , l_linearSolver(initLink("linearSolver", "Linear solver used by this component"))
{
}

void EulerExplicitSolver::solve(const core::ExecParams* params,
                                SReal dt,
                                sofa::core::MultiVecCoordId xResult,
                                sofa::core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
    {
        return;
    }

    SCOPED_TIMER("EulerExplicitSolve");

    // Create the vector and mechanical operations tools. These are used to execute special operations (multiplication,
    // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside the mechanical objects)
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );

    // Let the mechanical operations know that the current solver is explicit. This will be propagated back to the
    // force fields during the addForce and addKToMatrix phase. Force fields use this information to avoid
    // recomputing constant data in case of explicit solvers.
    mop->setImplicit(false);

    // Initialize the set of multi-vectors computed by this solver
    MultiVecDeriv acc   (&vop, core::vec_id::write_access::dx);     // acceleration to be computed
    MultiVecDeriv f     (&vop, core::vec_id::write_access::force ); // force to be computed

    acc.realloc(&vop, !d_threadSafeVisitor.getValue(), true);

    addSeparateGravity(&mop, dt, vResult);
    computeForce(&mop, f);

    sofa::Size nbNonDiagonalMasses = 0;
    MechanicalGetNonDiagonalMassesCountVisitor(&mop.mparams, &nbNonDiagonalMasses).execute(this->getContext());

    // Mass matrix is diagonal, solution can thus be found by computing acc = f/m
    if(nbNonDiagonalMasses == 0.)
    {
        // acc = M^-1 * f
        computeAcceleration(&mop, acc, f);
        projectResponse(&mop, acc);
        solveConstraints(&mop, acc);
    }
    else
    {
        projectResponse(&mop, f);

        if (l_linearSolver.get())
        {
            // Build the global matrix. In this solver, it is the global mass matrix
            // Projective constraints are also projected in this step
            assembleSystemMatrix(&mop);

            // Solve the system to find the acceleration
            // Solve M * a = f
            solveSystem(acc, f);
        }
        else
        {
            msg_error() << "Due to the presence of non-diagonal masses, the solver requires a linear solver";
            d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    // Compute the new position and new velocity from the acceleration
    updateState(&vop, &mop, xResult, vResult, acc, dt);
}

void EulerExplicitSolver::updateState(sofa::simulation::common::VectorOperations* vop,
                                      sofa::simulation::common::MechanicalOperations* mop,
                                      sofa::core::MultiVecCoordId xResult,
                                      sofa::core::MultiVecDerivId vResult,
                                      const sofa::core::behavior::MultiVecDeriv& acc,
                                      SReal dt) const
{
    SCOPED_TIMER("updateState");

    // Initialize the set of multi-vectors computed by this solver
    // "xResult" could be "position()" or "freePosition()" depending on the
    // animation loop calling this ODE solver.
    // Similarly, "vResult" could be "velocity()" or "freeVelocity()".
    // In case "xResult" refers to "position()", "newPos" refers the
    // same multi-vector than "pos". Similarly, for "newVel" and "vel".
    MultiVecCoord newPos(vop, xResult);                    // velocity to be computed
    MultiVecDeriv newVel(vop, vResult);                    // position to be computed

    // Initialize the set of multi-vectors used to compute the new velocity and position
    MultiVecCoord pos(vop, core::vec_id::write_access::position ); //current position
    MultiVecDeriv vel(vop, core::vec_id::write_access::velocity ); //current velocity

#ifdef SOFA_NO_VMULTIOP // unoptimized version
    if (d_symplectic.getValue())
    {
        //newVel = vec + acc * dt
        //newPos = pos + newVel * dt

        newVel.eq(vel, acc.id(), dt);
        mop->solveConstraint(newVel,core::ConstraintOrder::VEL);

        newPos.eq(pos, newVel, dt);
        mop->solveConstraint(newPos,core::ConstraintOrder::POS);
    }
    else
    {
        //newPos = pos + vel * dt
        //newVel = vel + acc * dt

        newPos.eq(pos, vel, dt);
        mop->solveConstraint(newPos,core::ConstraintOrder::POS);

        newVel.eq(vel, acc.id(), dt);
        mop->solveConstraint(newVel,core::ConstraintOrder::VEL);
    }
#else // single-operation optimization
    {
        typedef core::behavior::BaseMechanicalState::VMultiOp VMultiOp;

        // Create a set of linear operations that will be executed on two vectors
        // In our case, the operations will be executed to compute the new velocity vector,
        // and the new position vector. The order of execution is defined by
        // the symplectic property of the solver.
        VMultiOp ops(2);

        // Change order of operations depending on the symplectic flag
        const VMultiOp::size_type posId = d_symplectic.getValue(); // 1 if symplectic, 0 otherwise
        const VMultiOp::size_type velId = 1 - posId; // 0 if symplectic, 1 otherwise

        // Access the set of operations corresponding to the velocity vector
        // In case of symplectic solver, these operations are executed first.
        auto& ops_vel = ops[velId];

        // Associate the new velocity vector as the result to this set of operations
        ops_vel.first = newVel;

        // The two following operations are actually a unique operation: newVel = vel + dt * acc
        // The value 1.0 indicates that the first operation is based on the values
        // in the second pair and, therefore, the second operation is discarded.
        ops_vel.second.emplace_back(vel.id(), 1.0);
        ops_vel.second.emplace_back(acc.id(), dt);

        // Access the set of operations corresponding to the position vector
        // In case of symplectic solver, these operations are executed second.
        auto& ops_pos = ops[posId];

        // Associate the new position vector as the result to this set of operations
        ops_pos.first = newPos;

        // The two following operations are actually a unique operation: newPos = pos + dt * v
        // where v is "newVel" in case of a symplectic solver, and "vel" otherwise.
        // If symplectic: newPos = pos + dt * newVel, executed after newVel has been computed
        // If not symplectic: newPos = pos + dt * vel
        // The value 1.0 indicates that the first operation is based on the values
        // in the second pair and, therefore, the second operation is discarded.
        ops_pos.second.emplace_back(pos.id(), 1.0);
        ops_pos.second.emplace_back(d_symplectic.getValue() ? newVel.id() : vel.id(), dt);

        // Execute the defined operations to compute the new velocity vector and
        // the new position vector.
        // 1. Calls the "vMultiOp" method of every mapped BaseMechanicalState objects found in the
        // current context tree. This method may be called with different parameters than for the non-mapped
        // BaseMechanicalState objects.
        // 2. Calls the "vMultiOp" method of every BaseMechanicalState objects found in the
        // current context tree.
        vop->v_multiop(ops);

        // Calls "solveConstraint" on every ConstraintSolver objects found in the current context tree.
        mop->solveConstraint(newVel,core::ConstraintOrder::VEL);
        mop->solveConstraint(newPos,core::ConstraintOrder::POS);
    }
#endif
}

SReal EulerExplicitSolver::getIntegrationFactor(int inputDerivative, int outputDerivative) const
{
    if (inputDerivative >= 3 || outputDerivative >= 3)
    {
        return 0;
    }

    const SReal dt = getContext()->getDt();
    const SReal k_a = d_symplectic.getValue() * dt * dt;
    const SReal matrix[3][3] =
        {
                { 1, dt, k_a}, //x = 1 * x + dt * v + k_a * a
                { 0,  1,  dt}, //v = 0 * x +  1 * v +  dt * a
                { 0,  0,   0}
        };

    return matrix[outputDerivative][inputDerivative];
}

SReal EulerExplicitSolver::getSolutionIntegrationFactor(int outputDerivative) const
{
    if (outputDerivative >= 3)
        return 0;

    const SReal dt = getContext()->getDt();
    const SReal k_a = d_symplectic.getValue() * dt * dt;
    const SReal vect[3] = {k_a, dt, 1};
    return vect[outputDerivative];
}

void EulerExplicitSolver::init()
{
    OdeSolver::init();

    if (!l_linearSolver.get())
    {
        l_linearSolver.set(getContext()->get<LinearSolver>());
    }

    reinit();
    d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void EulerExplicitSolver::addSeparateGravity(sofa::simulation::common::MechanicalOperations* mop, SReal dt, core::MultiVecDerivId v)
{
    SCOPED_TIMER("addSeparateGravity");

    /// Calls the "addGravityToV" method of every BaseMass objects found in the current
    /// context tree, if the BaseMass object has the m_separateGravity flag set to true.
    /// The method "addGravityToV" usually performs v += dt * g
    mop->addSeparateGravity(dt, v);
}

void EulerExplicitSolver::computeForce(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId f)
{
    SCOPED_TIMER("ComputeForce");

    // 1. Clear the force vector (F := 0)
    // 2. Go down in the current context tree calling addForce on every forcefields
    // 3. Go up from the current context tree leaves calling applyJT on every mechanical mappings
    mop->computeForce(f);
}

void EulerExplicitSolver::computeAcceleration(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId acc, core::ConstMultiVecDerivId f)
{
    SCOPED_TIMER("AccFromF");

    // acc = M^-1 f
    // Since it requires the inverse of the mass matrix, this method is
    // probably implemented only for trivial matrix inversion, such as
    // a diagonal matrix.
    // For example, for a diagonal mass: a_i := f_i / M_ii
    mop->accFromF(acc, f);
}

void EulerExplicitSolver::projectResponse(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId vecId)
{
    SCOPED_TIMER("projectResponse");

    // Calls the "projectResponse" method of every BaseProjectiveConstraintSet objects found in the
    // current context tree. An example of such constraint set is the FixedProjectiveConstraint. In this case,
    // it will set to 0 every row (i, _) of the input vector for the ith degree of freedom.
    mop->projectResponse(vecId);
}

void EulerExplicitSolver::solveConstraints(sofa::simulation::common::MechanicalOperations* mop, core::MultiVecDerivId acc)
{
    SCOPED_TIMER("solveConstraint");

    // Calls "solveConstraint" method of every ConstraintSolver objects found in the current context tree.
    mop->solveConstraint(acc, core::ConstraintOrder::ACC);
}

void EulerExplicitSolver::assembleSystemMatrix(sofa::simulation::common::MechanicalOperations* mop) const
{
    SCOPED_TIMER("MBKBuild");

    //    A. For LinearSolver using a GraphScatteredMatrix (ie, non-assembled matrices), nothing appends.
    //    B. For LinearSolver using other type of matrices (FullMatrix, SparseMatrix, CompressedRowSparseMatrix),
    //       the "addMBKToMatrix" method is called on each BaseForceField objects and the "applyConstraint" method
    //       is called on every BaseProjectiveConstraintSet objects. An example of such constraint set is the
    //       FixedProjectiveConstraint. In this case, it will set to 0 every column (_, i) and row (i, _) of the assembled
    //       matrix for the ith degree of freedom.
    mop->setSystemMBKMatrix(
        core::MatricesFactors::M(1),
        core::MatricesFactors::B(0),
        core::MatricesFactors::K(0), l_linearSolver.get());
}

void EulerExplicitSolver::solveSystem(core::MultiVecDerivId solution, core::MultiVecDerivId rhs) const
{
    SCOPED_TIMER("MBKSolve");
    l_linearSolver->getLinearSystem()->setSystemSolution(solution);
    l_linearSolver->getLinearSystem()->setRHS(rhs);
    l_linearSolver->solveSystem();
    l_linearSolver->getLinearSystem()->dispatchSystemSolution(solution);
}

} // namespace sofa::component::odesolver::forward
