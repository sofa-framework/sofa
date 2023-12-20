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
#include <sofa/component/odesolver/backward/StaticSolver.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/core/behavior/MultiMatrix.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>

#include <iomanip>
#include <chrono>
#include <memory>

using sofa::simulation::mechanicalvisitor::MechanicalPropagateOnlyPositionAndVelocityVisitor;

namespace sofa::component::odesolver::backward
{

using sofa::core::VecId;
using namespace sofa::defaulttype;
using namespace sofa::core::behavior;

StaticSolver::StaticSolver()
    : d_newton_iterations(initData(&d_newton_iterations,
            (unsigned) 1,
            "newton_iterations",
            "Number of newton iterations between each load increments (normally, one load increment per simulation time-step."))
    , d_absolute_correction_tolerance_threshold(initData(&d_absolute_correction_tolerance_threshold,
            1e-5_sreal,
            "absolute_correction_tolerance_threshold",
            "Convergence criterion: The newton iterations will stop when the norm |du| is smaller than this threshold."))
    , d_relative_correction_tolerance_threshold(initData(&d_relative_correction_tolerance_threshold,
            1e-5_sreal,
            "relative_correction_tolerance_threshold",
            "Convergence criterion: The newton iterations will stop when the ratio |du| / |U| is smaller than this threshold."))
    , d_absolute_residual_tolerance_threshold( initData(&d_absolute_residual_tolerance_threshold,
            1e-5_sreal,
            "absolute_residual_tolerance_threshold",
            "Convergence criterion: The newton iterations will stop when the norm |R| is smaller than this threshold. "
            "Use a negative value to disable this criterion."))
    , d_relative_residual_tolerance_threshold( initData(&d_relative_residual_tolerance_threshold,
            1e-5_sreal,
            "relative_residual_tolerance_threshold",
            "Convergence criterion: The newton iterations will stop when the ratio |R|/|R0| is smaller than this threshold. "
            "Use a negative value to disable this criterion."))
    , d_should_diverge_when_residual_is_growing( initData(&d_should_diverge_when_residual_is_growing,
            false,
            "should_diverge_when_residual_is_growing",
            "Divergence criterion: The newton iterations will stop when the residual is greater than the one from the previous iteration."))
{}

void StaticSolver::solve(const sofa::core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    using namespace sofa::helper::logging;
    using namespace std::chrono;

    using std::chrono::steady_clock;
    using sofa::helper::ScopedAdvancedTimer;
    using sofa::core::behavior::MultiMatrix;
    using sofa::simulation::common::VectorOperations;
    using sofa::simulation::common::MechanicalOperations;

    static constexpr auto epsilon = std::numeric_limits<SReal>::epsilon();

    // Get the current context
    const auto context = this->getContext();

    // Create the vector and mechanical operations tools. These are used to execute special operations (multiplication,
    // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside the mechanical objects)
    VectorOperations vop( params, context );
    MechanicalOperations mop( params, context );

    // Initialize the set of multi-vectors used by this solver
    MultiVecCoord x(&vop, xResult );
    MultiVecDeriv force( &vop, sofa::core::VecDerivId::force() );
    MultiVecDeriv dx( &vop, sofa::core::VecDerivId::dx() );
    dx.realloc( &vop , true, true);
    U.realloc( &vop );
    U.clear();
    dx.clear();

    // Set the multi-vector identifier inside the mechanical parameters.
    sofa::core::MechanicalParams mechanical_parameters (*params);
    mechanical_parameters.setX(xResult);
    mechanical_parameters.setV(vResult);
    mechanical_parameters.setF(force);
    mechanical_parameters.setDx(dx);
    mechanical_parameters.setDt(dt);

    // Let the mechanical operations know that this is a non-linear solver. This will be propagated back to the
    // force fields during the addForce and addKToMatrix phase, which will let them recompute their internal
    // stresses if they have a non-linear relationship with the displacement.
    mop->setImplicit(true);

    // Options for the Newton-Raphson
    const auto & relative_correction_tolerance_threshold = d_relative_correction_tolerance_threshold.getValue();
    const auto & absolute_correction_tolerance_threshold = d_absolute_correction_tolerance_threshold.getValue();
    const auto & relative_residual_tolerance_threshold = d_relative_residual_tolerance_threshold.getValue();
    const auto & absolute_residual_tolerance_threshold = d_absolute_residual_tolerance_threshold.getValue();
    const auto & max_number_of_newton_iterations = d_newton_iterations.getValue();
    const auto & should_diverge_when_residual_is_growing = d_should_diverge_when_residual_is_growing.getValue();
    const auto & print_log = f_printLog.getValue();
    auto info = MessageDispatcher::info(Message::Runtime, std::make_shared<ComponentInfo>(this->getClassName()), SOFA_FILE_INFO);

    // Local variables used for the iterations
    unsigned n_it=0;
    double dx_squared_norm = 0, U_squared_norm = 0, R_squared_norm = 0, R0_squared_norm = 0, R_previous_squared_norm = 0;
    const auto absolute_squared_residual_threshold = absolute_residual_tolerance_threshold*absolute_residual_tolerance_threshold;
    const auto relative_squared_residual_tolerance_threshold = relative_residual_tolerance_threshold*relative_residual_tolerance_threshold;
    const auto absolute_squared_correction_threshold = absolute_correction_tolerance_threshold*absolute_correction_tolerance_threshold;
    const auto relative_squared_correction_threshold = relative_correction_tolerance_threshold*relative_correction_tolerance_threshold;
    bool converged = false, diverged = false;
    steady_clock::time_point t;

    // Reset the list of residual norms for this time step
    p_squared_residual_norms.clear();
    p_squared_residual_norms.reserve(max_number_of_newton_iterations);

    // Reset the list of increment norms for this time step
    p_squared_increment_norms.clear();
    p_squared_increment_norms.reserve(max_number_of_newton_iterations);

    if (print_log)
    {
        info << "======= Starting static ODE solver =======\n";
        info << "Time step                  : " << this->getTime() << "\n";
        info << "Context                    : " << dynamic_cast<const sofa::simulation::Node *>(context)->getPathName() << "\n";
        info << "Max number of iterations   : " << max_number_of_newton_iterations << "\n";
        info << "Residual tolerance (abs)   : " << absolute_residual_tolerance_threshold << "\n";
        info << "Residual tolerance (rel)   : " << relative_residual_tolerance_threshold << "\n";
        info << "Correction tolerance (abs) : " << absolute_correction_tolerance_threshold << "\n";
        info << "Correction tolerance (rel) : " << relative_correction_tolerance_threshold << "\n";
    }

    // Start the advanced timer
    SCOPED_TIMER("StaticSolver::Solve");

    // ###########################################################################
    // #                             First residual                              #
    // ###########################################################################
    // # Before starting any newton iterations, we first need to compute         #
    // # the residual with the updated right-hand side (the new load increment)  #
    // ###########################################################################
    {
        SCOPED_TIMER_VARNAME(computeForceTimer, "ComputeForce");

        // Step 1   Assemble the force vector
        // 1. Clear the force vector (F := 0)
        // 2. Go down in the current context tree calling addForce on every forcefields
        // 3. Go up from the current context tree leaves calling applyJT on every mechanical mappings
        mop.computeForce(force, true /* clear */);

        // Step 2   Projective constraints
        // Calls the "projectResponse" method of every BaseProjectiveConstraintSet objects found in the
        // current context tree. An example of such constraint set is the FixedProjectiveConstraint. In this case,
        // it will set to 0 every row (i, _) of the right-hand side (force) vector for the ith degree of
        // freedom.
        mop.projectResponse(force);
    }

    // Compute the initial residual
    R_squared_norm = force.dot(force);

    if (absolute_residual_tolerance_threshold > 0 && R_squared_norm <= absolute_squared_residual_threshold)
    {
        converged = true;
        if (print_log)
        {
            info << "The ODE has already reached an equilibrium state."
                 << std::scientific
                 << "The residual's ratio |R| is " << std::setw(12) << std::sqrt(R_squared_norm)
                 << " (criterion is " << std::setw(12) << absolute_residual_tolerance_threshold << ") \n"
                 << std::defaultfloat;
        }
    }

    // ###########################################################################
    // #                          Newton iterations                              #
    // ###########################################################################

    while (! converged && n_it < max_number_of_newton_iterations)
    {
        SCOPED_TIMER_VARNAME(step_timer, "NewtonStep");
        t = steady_clock::now();

        // Part I. Assemble the system matrix.
        MultiMatrix<MechanicalOperations> matrix(&mop);
        {
            SCOPED_TIMER("MBKBuild");
            // 1. The MechanicalMatrix::K is a simple structure that stores three floats called factors: m, b and k.
            // 2. the * operator simply multiplies each of the three factors with a value. No matrix is built yet.
            // 3. The = operator first search for a linear solver in the current context. It then calls the
            //    "setSystemMBKMatrix" method of the linear solver.

            //    A. For LinearSolver using a GraphScatteredMatrix (ie, non-assembled matrices), nothing appends.
            //    B. For LinearSolver using other type of matrices (FullMatrix, SparseMatrix, CompressedRowSparseMatrix),
            //       the "addMBKToMatrix" method is called on each BaseForceField objects and the "applyConstraint" method
            //       is called on every BaseProjectiveConstraintSet objects. An example of such constraint set is the
            //       FixedProjectiveConstraint. In this case, it will set to 0 every column (_, i) and row (i, _) of the assembled
            //       matrix for the ith degree of freedom.
            matrix.setSystemMBKMatrix(MechanicalMatrix::K * -1.0);
        }

        // Part II. Solve the unknown increment.
        {
            SCOPED_TIMER("MBKSolve");
            // Calls methods "setSystemRHVector", "setSystemLHVector" and "solveSystem" of the LinearSolver component
            // for CG: calls iteratively addDForce, mapped:  [applyJ, addDForce, applyJt(vec)]+
            // for Direct: solves the system, everything's already assembled
            matrix.solve(dx, force);
        }

        // Part III. Propagate the solution increment and update geometry.
        {
            SCOPED_TIMER("PropagateDx");
            // Updating the geometry
            x.peq(dx); // x := x + dx

            // Calls "solveConstraint" method of every ConstraintSolver objects found in the current context tree.
            // todo(jnbrunet): Shouldn't this be done AFTER the position propagation of the mapped nodes?
            mop.solveConstraint(x, sofa::core::ConstraintOrder::POS);

            // Propagate positions to mapped mechanical objects, for example, identity mappings, barycentric mappings, ...
            // This will call the methods apply and applyJ on every mechanical mappings.
            MechanicalPropagateOnlyPositionAndVelocityVisitor(&mechanical_parameters).execute(context);
        }

        // At this point, we completed one iteration, increment the counter.
        // The rest is only for convergence tests and logging.
        n_it++;

        // The rest of the step is only necessary when doing more than one Newton iteration. Otherwise, we will
        // waste computation time to reassemble the residual and compute the norms for a convergence that will
        // never happen (we will always reach the maximum number of iterations, which is 1)
        if (max_number_of_newton_iterations == 1)
        {
            converged = true; // Not really, but we won't warn about divergence when it is always the case
            diverged = false;
            break;
        }

        // Part IV. Update the force vector.
        {
            SCOPED_TIMER("UpdateForce");

            mop.computeForce(force);
            mop.projectResponse(force);
        }

        // Part V. Compute the updated norms.
        {
            SCOPED_TIMER("ComputeNorms");

            // Residual norm
            R_squared_norm = force.dot(force);

            if (n_it == 1)
            {
                R0_squared_norm = R_squared_norm;
                R_previous_squared_norm = R0_squared_norm;
            }

            p_squared_residual_norms.emplace_back(R_squared_norm);

            // Displacement norm
            U.peq(dx);
            dx_squared_norm = dx.dot(dx);
            U_squared_norm= U.dot(U);

            p_squared_increment_norms.emplace_back(dx_squared_norm);
        }

        // Part VI. Stop timers and print step information.
        {
            auto iteration_time = duration_cast<nanoseconds>(steady_clock::now() - t).count();

            if (print_log)
            {
                info << "Newton iteration #" << std::left << std::setw(5) << n_it
                     << std::scientific
                     << "  |R| = "        << std::setw(12) << std::sqrt(R_squared_norm)
                     << "  |R|/|R0| = "   << std::setw(12) << (R0_squared_norm < epsilon*epsilon ? 0 : std::sqrt(R_squared_norm / R0_squared_norm))
                     << "  |du| = "       << std::setw(12) << std::sqrt(dx_squared_norm)
                     << "  |du| / |U| = " << std::setw(12) << (U_squared_norm < epsilon*epsilon  ? 0 : std::sqrt(dx_squared_norm / U_squared_norm))
                     << std::defaultfloat;
                info << "  Time = " << iteration_time / 1000 / 1000 << " ms";
                info << "\n";
            }
        }

        // Part VII. Check for convergence/divergence
        {
            if (std::isnan(R_squared_norm) || std::isnan(dx_squared_norm) || U_squared_norm < epsilon*epsilon)
            {
                diverged = true;
                if (print_log)
                {
                    info << "[DIVERGED]";
                    if (std::isnan(R_squared_norm))
                    {
                        info << " The residual's ratio |R| is NaN.";
                    }
                    if (std::isnan(dx_squared_norm))
                    {
                        info << " The correction's ratio |du| is NaN.";
                    }
                    if (U_squared_norm < epsilon)
                    {
                        info << " The correction's ratio |du|/|U| is NaN (|U| is zero).";
                    }
                    info << "\n";
                }
                break;
            }

            if (absolute_correction_tolerance_threshold > 0 && dx_squared_norm < absolute_squared_correction_threshold)
            {
                converged = true;
                if (print_log)
                {
                    info  << "[CONVERGED] The correction's norm |du| = " << std::sqrt(dx_squared_norm) << " is smaller than the threshold of " << absolute_correction_tolerance_threshold << ".\n";
                }
                break;
            }

            if (relative_correction_tolerance_threshold > 0 && dx_squared_norm < relative_squared_correction_threshold*U_squared_norm)
            {
                converged = true;
                if (print_log)
                {
                    info  << "[CONVERGED] The correction's ratio |du|/|U| = " << std::sqrt(dx_squared_norm/U_squared_norm) << " is smaller than the threshold of " << relative_correction_tolerance_threshold << ".\n";
                }
                break;
            }

            if (absolute_residual_tolerance_threshold > 0 && R_squared_norm < absolute_squared_residual_threshold)
            {
                converged = true;
                if (print_log)
                {
                    info << "[CONVERGED] The residual's norm |R| = " << std::sqrt(R_squared_norm) << " is smaller than the threshold of " << absolute_residual_tolerance_threshold << ".\n";
                }
                break;
            }

            if (relative_residual_tolerance_threshold > 0 && R_squared_norm < relative_squared_residual_tolerance_threshold*R0_squared_norm)
            {
                converged = true;
                if (print_log)
                {
                    info << "[CONVERGED] The residual's ratio |R|/|R0| = " << std::sqrt(R_squared_norm/R0_squared_norm) << " is smaller than the threshold of " << relative_residual_tolerance_threshold << ".\n";
                }
                break;
            }

            if (should_diverge_when_residual_is_growing && R_squared_norm > R_previous_squared_norm)
            {
                diverged = true;
                if (print_log)
                {
                    info << "[DIVERGED] The current residual norm |R| = " << std::sqrt(R_squared_norm)
                         << " is greater than at the previous Newton iteration (" << std::sqrt(R_previous_squared_norm) << ").\n";
                }
            }
        }

        // This is used to detect a rise of residual (divergence test)
        R_previous_squared_norm = R_squared_norm;
    }

    n_it--; // Reset to the actual index of the last iteration completed

    if (! converged && ! diverged && n_it == (max_number_of_newton_iterations-1))
    {
        if (print_log)
        {
            info << "[DIVERGED] The number of Newton iterations reached the maximum of " << max_number_of_newton_iterations << " iterations" << ".\n";
        }
    }

    sofa::helper::AdvancedTimer::valSet("nb_iterations", n_it+1);
    sofa::helper::AdvancedTimer::valSet("residual", std::sqrt(R_squared_norm));
    sofa::helper::AdvancedTimer::valSet("correction", std::sqrt(dx_squared_norm));
}


int StaticSolverClass = sofa::core::RegisterObject("Static ODE Solver")
    .add< StaticSolver >()
;

} // namespace sofa::component::odesolver::backward
