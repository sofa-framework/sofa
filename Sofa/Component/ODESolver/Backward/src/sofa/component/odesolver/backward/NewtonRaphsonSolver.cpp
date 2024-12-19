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
#include <sofa/component/odesolver/backward/NewtonRaphsonSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>


namespace sofa::component::odesolver::backward
{

NewtonRaphsonSolver::NewtonRaphsonSolver()
    : l_integrationMethod(initLink("integrationMethod", "The integration method to use in a Newton iteration"))
    , d_maxNbIterationsNewton(initData(&d_maxNbIterationsNewton, 1u, "maxNbIterationsNewton",
        "Maximum number of iterations of the Newton's method if it has not converged."))
    , d_absoluteResidualToleranceThreshold(initData(&d_absoluteResidualToleranceThreshold, 1e-9_sreal,
        "absoluteResidualToleranceThreshold",
        "The newton iterations will stop when the norm of the residual at "
        "iteration k is lower than this threshold."))
    , d_maxNbIterationsLineSearch(initData(&d_maxNbIterationsLineSearch, 5u, "maxNbIterationsLineSearch",
        "Maximum number of iterations of the line search method if it has not converged."))
    , d_lineSearchCoefficient(initData(&d_lineSearchCoefficient, 0.5_sreal, "lineSearchCoefficient", "Line search coefficient"))
{}

void NewtonRaphsonSolver::init()
{
    OdeSolver::init();
    LinearSolverAccessor::init();

    if (!l_integrationMethod.get())
    {
        l_integrationMethod.set(getContext()->get<core::behavior::BaseIntegrationMethod>(getContext()->getTags(), core::objectmodel::BaseContext::SearchDown));

        if (!l_integrationMethod)
        {
            msg_error() << "An integration method is required by this component but has not been found.";
            this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
            return;
        }
    }

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void NewtonRaphsonSolver::computeRightHandSide(const core::ExecParams* params, SReal dt,
                                       core::MultiVecDerivId force,
                                       core::MultiVecDerivId b,
                                       core::MultiVecDerivId velocity_i,
                                       core::MultiVecCoordId position_i) const
{
    core::behavior::RHSInput input;
    input.intermediateVelocity = velocity_i;
    input.intermediatePosition = position_i;

    l_integrationMethod->computeRightHandSide(params, input, force, b, dt);
}

SReal NewtonRaphsonSolver::computeResidual(const core::ExecParams* params, sofa::simulation::common::MechanicalOperations& mop,
    SReal dt, core::MultiVecDerivId force,
    core::MultiVecDerivId oldVelocity,
    core::MultiVecDerivId newVelocity)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );

    core::behavior::MultiVecDeriv residual(&vop, true, core::VecIdProperties{"residual", GetClass()->className});

    core::behavior::MultiVecDeriv tmp(&vop);

    vop.v_eq(tmp, newVelocity);
    vop.v_peq(tmp, oldVelocity, -1);
    mop.addMdx(residual, tmp);

    vop.v_peq(residual, force, -dt);

    vop.v_dot(residual, residual);
    return vop.finish();
}

void NewtonRaphsonSolver::solve(
    const core::ExecParams* params, SReal dt,
    sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
    {
        return;
    }

    const bool printLog = f_printLog.getValue();

    // Create the vector and mechanical operations tools. These are used to execute special operations (multiplication,
    // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside the mechanical objects)
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(true);

    //force vector that will be computed in this solve. The content of the
    //previous time step will be erased.
    core::behavior::MultiVecDeriv force(&vop, core::vec_id::write_access::force );
    core::behavior::MultiVecDeriv b(&vop, true, core::VecIdProperties{"RHS", GetClass()->className});

    core::behavior::MultiVecDeriv velocityPrevious(&vop);
    velocityPrevious.eq(vResult);

    //the intermediate position and velocity required by the Newton's algorithm
    core::behavior::MultiVecCoord position_i(&vop);
    core::behavior::MultiVecDeriv velocity_i(&vop);

    //initial guess: the new states are initialized with states from the previous time step
    position_i.eq(xResult);
    velocity_i.eq(vResult);

    //the position and velocity that will be computed at the end of this algorithm
    core::behavior::MultiVecCoord newPosition(&vop, xResult );
    core::behavior::MultiVecDeriv newVelocity(&vop, vResult );

    //dx vector is required by some operations of the algorithm, even if it is
    //not explicit
    core::behavior::MultiVecDeriv dx(&vop, sofa::core::vec_id::write_access::dx);
    dx.realloc(&vop, false, true);

    m_linearSystemSolution.realloc(&vop, false, true,
        core::VecIdProperties{"solution", GetClass()->className});

    // inform the constraint parameters about the position and velocity id
    mop.cparams.setX(xResult);
    mop.cparams.setV(vResult);

    l_integrationMethod->initializeVectors(params,
        core::vec_id::read_access::position,
        core::vec_id::read_access::velocity);

    {
        SCOPED_TIMER("ComputeRHS");
        computeRightHandSide(params, dt, force, b, velocity_i, position_i);
    }

    SReal squaredResidualNorm{};
    {
        SCOPED_TIMER("ComputeError");
        squaredResidualNorm = this->computeResidual(params, mop, dt, force, velocityPrevious, velocity_i);

        msg_info() << "The initial residual squared norm is " << squaredResidualNorm << ". ";
    }

    const auto squaredAbsoluteResidualToleranceThreshold = std::pow(d_absoluteResidualToleranceThreshold.getValue(), 2);

    if (squaredResidualNorm <= squaredAbsoluteResidualToleranceThreshold)
    {
        msg_info() << "The ODE has already reached an equilibrium state. "
            << "The residual squared norm is " << squaredResidualNorm << ". "
            << "The threshold for convergence is " << squaredAbsoluteResidualToleranceThreshold;
    }
    else
    {
        SCOPED_TIMER("NewtonsIterations");

        const auto maxNbIterationsNewton = d_maxNbIterationsNewton.getValue();
        const auto maxNbIterationsLineSearch = d_maxNbIterationsLineSearch.getValue();
        const auto [mFact, bFact, kFact] = l_integrationMethod->getMatricesFactors(dt);
        bool hasConverged = false;
        const auto lineSearchCoefficient = d_lineSearchCoefficient.getValue();

        unsigned int newtonIterationCount = 0;
        for (; newtonIterationCount < maxNbIterationsNewton; ++newtonIterationCount)
        {
            //assemble the system matrix
            {
                SCOPED_TIMER("setSystemMBKMatrix");
                mop.setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());
            }

            //solve the system
            {
                SCOPED_TIMER("MBKSolve");

                l_linearSolver->setSystemLHVector(m_linearSystemSolution);
                l_linearSolver->setSystemRHVector(b);
                l_linearSolver->solveSystem();
            }

            bool lineSearchSuccess = false;
            SReal squaredResidualNormLineSearch {};
            SReal totalLineSearchCoefficient { 1_sreal };
            SReal minTotalLineSearchCoefficient { 1_sreal };
            SReal minSquaredResidualNormLineSearch { std::numeric_limits<SReal>::max() };

            const auto lineSearch = [&](bool applyCoefficient)
            {
                if (applyCoefficient)
                {
                    // solution *= lineSearchCoefficient
                    vop.v_teq(m_linearSystemSolution, lineSearchCoefficient);
                    totalLineSearchCoefficient *= lineSearchCoefficient;
                }

                l_integrationMethod->updateStates(params, dt,
                    position_i, velocity_i,
                    newPosition, newVelocity,
                    m_linearSystemSolution);

                mop.projectPositionAndVelocity(newPosition, newVelocity);
                mop.propagateXAndV(newPosition, newVelocity);

                computeRightHandSide(params, dt, force, b, newVelocity, newPosition);

                squaredResidualNormLineSearch = this->computeResidual(params, mop, dt, force, velocityPrevious, newVelocity);
                if (squaredResidualNormLineSearch < minSquaredResidualNormLineSearch)
                {
                    minSquaredResidualNormLineSearch = squaredResidualNormLineSearch;
                    minTotalLineSearchCoefficient = totalLineSearchCoefficient;
                }
                return squaredResidualNormLineSearch < squaredResidualNorm;
            };

            for (unsigned int lineSearchIterationCount = 0; lineSearchIterationCount < maxNbIterationsLineSearch; ++lineSearchIterationCount)
            {
                if (lineSearch(lineSearchIterationCount > 0))
                {
                    lineSearchSuccess = true;
                    break;
                }
            }

            if (!lineSearchSuccess)
            {
                msg_warning_when(maxNbIterationsLineSearch > 0) << "Line search failed at Newton iteration "
                    << newtonIterationCount << ". Using the coefficient "
                    << minTotalLineSearchCoefficient << " resulting to the minimal residual norm (" << minSquaredResidualNormLineSearch << ").";

                vop.v_teq(m_linearSystemSolution, minTotalLineSearchCoefficient / totalLineSearchCoefficient);
                lineSearch(false);
            }

            squaredResidualNorm = squaredResidualNormLineSearch;

            vop.v_eq(position_i, newPosition);
            vop.v_eq(velocity_i, newVelocity);

            if (squaredResidualNorm <= squaredAbsoluteResidualToleranceThreshold)
            {
                msg_info() << "[CONVERGED] residual squared norm (" <<
                        squaredResidualNorm << ") is smaller than the threshold ("
                        << squaredAbsoluteResidualToleranceThreshold << ") after "
                        << (newtonIterationCount+1) << " iterations. ";
                hasConverged = true;
                break;
            }
        }

        if (!hasConverged)
        {
            msg_warning() << "Newton's method failed to converge after " << newtonIterationCount
                << " iterations with residual squared norm = " << squaredResidualNorm << ". ";
        }
    }
}

void registerNewtonRaphsonSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Generic Newton-Raphson algorithm.")
        .add< NewtonRaphsonSolver >());
}

}
