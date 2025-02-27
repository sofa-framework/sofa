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
#include <sofa/component/odesolver/backward/NewtonRaphsonSolverBackup.h>
#include <sofa/component/odesolver/backward/convergence/AbsoluteConvergenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/AbsoluteEstimateDifferenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/RelativeEstimateDifferenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/RelativeInitialConvergenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/RelativeSuccessiveConvergenceMeasure.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

#include <iomanip>

namespace sofa::component::odesolver::backward
{

static constexpr NewtonStatus defaultStatus("Undefined");

NewtonRaphsonSolverBackup::NewtonRaphsonSolverBackup()
    : l_integrationMethod(initLink("integrationMethod", "The integration method to use in a Newton iteration"))
    , d_maxNbIterationsNewton(initData(&d_maxNbIterationsNewton, 1u, "maxNbIterationsNewton",
        "Maximum number of iterations of the Newton's method if it has not converged."))
    , d_relativeSuccessiveStoppingThreshold(initData(&d_relativeSuccessiveStoppingThreshold, 1e-5_sreal,
        "relativeSuccessiveStoppingThreshold",
        "Threshold for the relative successive progress criterion. The Newton "
        "iterations will stop when the ratio between the norm of the residual "
        "at iteration k over the norm of the residual at iteration k-1 is lower"
        " than this threshold."))
    , d_relativeInitialStoppingThreshold(initData(&d_relativeInitialStoppingThreshold, 1e-5_sreal,
        "relativeInitialStoppingThreshold",
        "Threshold for the relative initial progress criterion. The Newton"
        " iterations will stop when the ratio between the norm of the residual "
        "at iteration k over the norm of the residual at iteration 0 is"
        " lower than this threshold. This criterion tracks the overall progress "
        "made since the beginning of the iteration process. If the ratio is "
        "significantly smaller than 1, it indicates that the iterative process "
        "is making substantial progress, and the method is converging toward the"
        " root."))
    , d_absoluteResidualStoppingThreshold(initData(&d_absoluteResidualStoppingThreshold, 1e-5_sreal,
        "absoluteResidualStoppingThreshold",
        "Threshold for the absolute function value stopping criterion. The "
        "Newton iterations will stop when the norm of the residual at iteration "
        "k is lower than this threshold. This criterion indicates the current "
        "iteration found an estimate close to the root."))
    , d_relativeEstimateDifferenceThreshold(initData(&d_relativeEstimateDifferenceThreshold, 0_sreal,
        "relativeEstimateDifferenceThreshold",
        "Threshold for the relative change in root estimate criterion. The "
        "Newton iterations will stop when the difference between two successive "
        "estimates divided by the previous estimate is smaller than this threshold"))
    , d_absoluteEstimateDifferenceThreshold(initData(&d_absoluteEstimateDifferenceThreshold, 0_sreal,
        "absoluteEstimateDifferenceThreshold",
        "Threshold for the absolute change in root estimate criterion. The "
        "Newton iterations will stop when the difference between two successive "
        "estimates is smaller than this threshold."))
    , d_maxNbIterationsLineSearch(initData(&d_maxNbIterationsLineSearch, 5u, "maxNbIterationsLineSearch",
        "Maximum number of iterations of the line search method if it has not converged."))
    , d_lineSearchCoefficient(initData(&d_lineSearchCoefficient, 0.5_sreal, "lineSearchCoefficient", "Line search coefficient"))
    , d_updateStateWhenDiverged(initData(&d_updateStateWhenDiverged, true, "updateStateWhenDiverged", "Update the states within the last iteration even if the iterative process is considered diverged."))
    , d_status(initData(&d_status, defaultStatus, "status", ("status\n" + NewtonStatus::dataDescription()).c_str()))
    , d_residualGraph(initData(&d_residualGraph, "residualGraph", "Graph of the residual over the iterations"))
{
    d_status.setReadOnly(true);

    static std::string groupAnalysis { "Analysis" };
    d_status.setGroup(groupAnalysis);
    d_residualGraph.setGroup(groupAnalysis);

    d_residualGraph.setWidget("graph");
}

NewtonRaphsonSolverBackup::~NewtonRaphsonSolverBackup()
{
    sofa::simulation::common::VectorOperations vop(core::ExecParams::defaultInstance(), this->getContext());

    m_coordStates.setOps(&vop);
    m_derivStates.setOps(&vop);

    m_coordStates.newtonIterationStates.clear();
    m_derivStates.newtonIterationStates.clear();
    m_coordStates.timeStepStates.clear();
    m_derivStates.timeStepStates.clear();
}

void NewtonRaphsonSolverBackup::init()
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

    d_status.setValue(defaultStatus);

    this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
}

void NewtonRaphsonSolverBackup::reset()
{
    d_status.setValue(defaultStatus);

    auto graph = sofa::helper::getWriteAccessor(d_residualGraph);
    graph->clear();
}

void NewtonRaphsonSolverBackup::computeRightHandSide(const core::ExecParams* params, SReal dt,
                                               core::MultiVecDerivId force,
                                               core::MultiVecDerivId b,
                                               core::MultiVecDerivId velocity_i,
                                               core::MultiVecCoordId position_i) const
{
    core::behavior::RHSInput input;
    input.intermediateVelocity = velocity_i;
    input.intermediatePosition = position_i;

    l_integrationMethod->computeRightHandSide(
        params, input, force, b, dt);
}

SReal NewtonRaphsonSolverBackup::computeResidual(const core::ExecParams* params,
                                           sofa::simulation::common::MechanicalOperations& mop,
                                           SReal dt, core::MultiVecDerivId force,
                                           core::MultiVecDerivId oldVelocity,
                                           core::MultiVecDerivId newVelocity)
{
    return l_integrationMethod->computeResidual(params, dt, force, oldVelocity, newVelocity);
}

void NewtonRaphsonSolverBackup::resizeStateList(const std::size_t nbStates,
                                          sofa::simulation::common::VectorOperations& vop)
{
    if (nbStates < 1)
    {
        msg_error() << "The number of states should be >= 1";
    }
    const auto resizeState = [&vop, nbStates](auto& states, std::size_t newSize)
    {
        for (std::size_t i = states.size(); i < newSize; ++i)
        {
            states.emplace_back(&vop, true);
        }
    };

    resizeState(m_coordStates.timeStepStates, nbStates);
    resizeState(m_derivStates.timeStepStates, nbStates);
    resizeState(m_coordStates.newtonIterationStates, 2);
    resizeState(m_derivStates.newtonIterationStates, 2);
}

void NewtonRaphsonSolverBackup::start()
{
    // The status of the algorithm is set to "Running", and will be changed later
    // depending on the convergence of the algorithm.
    static constexpr auto running = NewtonStatus("Running");
    d_status.setValue(running);
}
bool NewtonRaphsonSolverBackup::measureConvergence(
    const NewtonRaphsonConvergenceMeasure& measure,
    sofa::simulation::common::VectorOperations& vop,
    sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult,
    std::stringstream& os)
{
    if (measure.isMeasured())
    {
        if (measure.hasConverged())
        {
            d_status.setValue(measure.status());

            msg_info() << os.str();
            msg_info() << "[CONVERGED] " << measure.writeWhenConverged();

            //algorithm has converged, states can be updated
            const NewtonIterationStateVersionAccess i;
            vop.v_eq(xResult, m_coordStates[i+1]);
            vop.v_eq(vResult, m_derivStates[i+1]);

            return true;
        }
        else if (notMuted())
        {
            os << "\n* " << measure.measureName() << ": NO";
        }
    }
    else if (notMuted())
    {
        os << "\n* " << measure.measureName() << ": NOT TESTED";
    }
    return false;
}

void NewtonRaphsonSolverBackup::solve(
    const core::ExecParams* params, SReal dt,
    sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    if (!isComponentStateValid())
    {
        return;
    }

    start();

    const bool printLog = f_printLog.getValue();
    auto graphAccessor = sofa::helper::getWriteAccessor(d_residualGraph);
    auto& graph = graphAccessor.wref();

    // Create the vector and mechanical operations tools. These are used to execute special
    // operations (multiplication,
    // additions, etc.) on multi-vectors (a vector that is stored in different buffers inside the mechanical objects)
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(true);

    //force vector that will be computed in this solve. The content of the
    //previous time step will be erased.
    core::behavior::MultiVecDeriv force(&vop, core::vec_id::write_access::force );
    core::behavior::MultiVecDeriv rhs(&vop, true, core::VecIdProperties{"RHS", GetClass()->className});

    //Number of previous steps required by the integration method
    //ex: Backward Euler is 1-step
    const auto stepSize = this->l_integrationMethod->stepSize();

    if (stepSize == 0)
    {
        msg_error() << "Step size must be greater than zero.";
        return;
    }
    
    resizeStateList(stepSize, vop);
    
    auto& x = m_coordStates;
    auto& v = m_derivStates;

    x.setOps(&vop);
    v.setOps(&vop);

    //This variable eases the reading of the code compared to the equations. For example, $x_{n+1}$ will be retrieved using x[n+1]
    // n is the state at the previous time step
    // n+1 is the state to be computed at the current time-step
    constexpr TimeStepStateVersionAccess n;

    //This variable eases the reading of the code compared to the equations. For example, $x_i$ will be retrieved using x[i]
    const NewtonIterationStateVersionAccess i;

    //initial guess: the new states are initialized with states from the previous time step
    x[i].eq(xResult);
    v[i].eq(vResult);

    //the position and velocity that will be computed at the end of this algorithm
    v[n].eq(vResult);
    x[n].eq(xResult);

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
        x[n],
        v[n]);

    {
        SCOPED_TIMER("ComputeRHS");
        computeRightHandSide(params, dt, force, rhs, v[i], x[i]);
        // msg_info() << "force 1 = " << force;
    }

    auto& residualList = graph["residual"];
    residualList.clear();

    SReal squaredResidualNorm{};
    {
        SCOPED_TIMER("ComputeError");
        squaredResidualNorm = this->computeResidual(
            params, mop, dt, force, v[n], v[i]);
        residualList.push_back(squaredResidualNorm);
    }

    const auto absoluteStoppingThreshold = d_absoluteResidualStoppingThreshold.getValue();
    const auto squaredAbsoluteStoppingThreshold = std::pow(absoluteStoppingThreshold, 2);

    if (absoluteStoppingThreshold > 0 && squaredResidualNorm <= squaredAbsoluteStoppingThreshold)
    {
        msg_info() << "The ODE has already reached an equilibrium state. "
            << "The residual squared norm is " << squaredResidualNorm << ". "
            << "The threshold for convergence is " << squaredAbsoluteStoppingThreshold;
        static constexpr auto convergedEquilibrium = NewtonStatus("ConvergedEquilibrium");
        d_status.setValue(convergedEquilibrium);
    }
    else
    {
        SCOPED_TIMER("NewtonsIterations");

        msg_info() << "Initial residual: " << squaredResidualNorm;

        const auto relativeSuccessiveStoppingThreshold = d_relativeSuccessiveStoppingThreshold.getValue();

        RelativeSuccessiveConvergenceMeasure relativeSuccessiveConvergenceMeasure(d_relativeSuccessiveStoppingThreshold.getValue());
        RelativeInitialConvergenceMeasure relativeInitialConvergenceMeasure(d_relativeInitialStoppingThreshold.getValue());
        relativeInitialConvergenceMeasure.firstSquaredResidualNorm = squaredResidualNorm;
        AbsoluteConvergenceMeasure absoluteConvergenceMeasure(absoluteStoppingThreshold);
        AbsoluteEstimateDifferenceMeasure absoluteEstimateDifferenceMeasure(d_absoluteEstimateDifferenceThreshold.getValue());
        RelativeEstimateDifferenceMeasure relativeEstimateDifferenceMeasure(d_relativeEstimateDifferenceThreshold.getValue());

        const auto maxNbIterationsNewton = d_maxNbIterationsNewton.getValue();
        const auto maxNbIterationsLineSearch = d_maxNbIterationsLineSearch.getValue();
        const auto [mFact, bFact, kFact] = l_integrationMethod->getMatricesFactors(dt);
        bool hasConverged = false;
        bool hasLineSearchFailed = false;
        const auto lineSearchCoefficient = d_lineSearchCoefficient.getValue();

        unsigned int newtonIterationCount = 0;
        for (; newtonIterationCount < maxNbIterationsNewton; ++newtonIterationCount)
        {
            msg_info() << "Newton iteration #" << newtonIterationCount;
            
            //assemble the system matrix
            {
                SCOPED_TIMER("setSystemMBKMatrix");
                mop.setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());
            }

            //solve the system
            {
                SCOPED_TIMER("MBKSolve");

                l_linearSolver->setSystemLHVector(m_linearSystemSolution);
                l_linearSolver->setSystemRHVector(rhs);
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
                    x[n], v[i],
                    x[i+1], v[i+1],
                    m_linearSystemSolution);
                // msg_info() << "solution = " << m_linearSystemSolution;
                // msg_info() << "x[n] = " << x[n];
                // msg_info() << "x[i+1] = " << x[i+1];

                mop.projectPositionAndVelocity(x[i+1], v[i+1]);
                mop.propagateXAndV(x[i+1], v[i+1]);

                computeRightHandSide(params, dt, force, rhs, v[i+1], x[i+1]);
                // msg_info() << "force = " << force;

                squaredResidualNormLineSearch = this->computeResidual(params, mop, dt, force, v[n], v[i+1]);
                msg_info() << "Squared residual norm: " << squaredResidualNormLineSearch;
                if (squaredResidualNormLineSearch < minSquaredResidualNormLineSearch)
                {
                    minSquaredResidualNormLineSearch = squaredResidualNormLineSearch;
                    minTotalLineSearchCoefficient = totalLineSearchCoefficient;
                }
                if (squaredResidualNormLineSearch < squaredResidualNorm)
                {
                    return true;
                }
                msg_info() << "Line search iteration failed (" << squaredResidualNormLineSearch
                           << " >= " << squaredResidualNorm << ")";
                return false;
            };

            unsigned int lineSearchIterationCount = 0;
            for (; lineSearchIterationCount < maxNbIterationsLineSearch; ++lineSearchIterationCount)
            {
                msg_info() << "Line search iteration #" << lineSearchIterationCount;
                if (lineSearch(lineSearchIterationCount > 0))
                {
                    lineSearchSuccess = true;
                    break;
                }
            }

            if (!lineSearchSuccess)
            {
                hasLineSearchFailed = true;
                msg_warning_when(maxNbIterationsLineSearch > 0) << "Line search failed at Newton iteration "
                    << newtonIterationCount << ". Using the coefficient "
                    << minTotalLineSearchCoefficient << " resulting to the minimal residual norm (" << minSquaredResidualNormLineSearch << "). Stopping the iterative process.";

                vop.v_teq(m_linearSystemSolution, minTotalLineSearchCoefficient / totalLineSearchCoefficient);
                lineSearch(false);
            }
            else
            {
                msg_info() << "Line search succeeded after " << lineSearchIterationCount << " iterations";
            }

            const auto previousSquaredResidualNorm = squaredResidualNorm;
            squaredResidualNorm = squaredResidualNormLineSearch;

            std::stringstream iterationResults;

            if (printLog)
            {
                iterationResults << "Newton iteration results:";
                iterationResults << "\n* Current iteration = " << newtonIterationCount;
                iterationResults << "\n* Residual = " << std::sqrt(squaredResidualNorm) << " (threshold = " << absoluteStoppingThreshold << ")";
                iterationResults << "\n* Successive relative ratio = " << std::sqrt(squaredResidualNorm / previousSquaredResidualNorm) << " (threshold = " << relativeSuccessiveStoppingThreshold << ", previous residual = " << std::sqrt(previousSquaredResidualNorm) << ")";
            }

            if (!lineSearchSuccess)
            {
                msg_info() << iterationResults.str();

                static constexpr auto divergedLineSearch = NewtonStatus("DivergedLineSearch");
                d_status.setValue(divergedLineSearch);

                break;
            }

            residualList.push_back(squaredResidualNorm);

            relativeSuccessiveConvergenceMeasure.squaredResidualNorm = squaredResidualNorm;
            relativeSuccessiveConvergenceMeasure.previousSquaredResidualNorm = previousSquaredResidualNorm;
            relativeSuccessiveConvergenceMeasure.newtonIterationCount = newtonIterationCount;
            if (measureConvergence(relativeSuccessiveConvergenceMeasure, vop, xResult, vResult, iterationResults))
            {
                hasConverged = true;
                break;
            }

            relativeInitialConvergenceMeasure.squaredResidualNorm = squaredResidualNorm;
            if (measureConvergence(relativeInitialConvergenceMeasure, vop, xResult, vResult, iterationResults))
            {
                hasConverged = true;
                break;
            }

            absoluteConvergenceMeasure.squaredResidualNorm = squaredResidualNorm;
            if (measureConvergence(absoluteConvergenceMeasure, vop, xResult, vResult, iterationResults))
            {
                hasConverged = true;
                break;
            }

            if (absoluteEstimateDifferenceMeasure.isMeasured()
                || relativeEstimateDifferenceMeasure.isMeasured())
            {
                {
                    core::behavior::MultiVecDeriv tmp(&vop);

                    vop.v_eq(tmp, v[i+1]);
                    vop.v_peq(tmp, v[i], -1);

                    vop.v_dot(tmp, tmp);
                }
                const auto squaredAbsoluteDifference = vop.finish();

                if (printLog)
                {
                    iterationResults << "\n* Successive estimate difference = " << std::sqrt(squaredAbsoluteDifference);
                }

                if (relativeEstimateDifferenceMeasure.isMeasured())
                {
                    vop.v_dot(v[i], v[i]);
                    relativeEstimateDifferenceMeasure.squaredAbsoluteDifference = squaredAbsoluteDifference;
                    relativeEstimateDifferenceMeasure.squaredPreviousVelocity = vop.finish();
                    
                    if (measureConvergence(relativeEstimateDifferenceMeasure, vop, xResult, vResult, iterationResults))
                    {
                        hasConverged = true;
                        break;
                    }
                }

                absoluteEstimateDifferenceMeasure.squaredAbsoluteDifference = squaredAbsoluteDifference;
                if (measureConvergence(absoluteEstimateDifferenceMeasure, vop, xResult, vResult, iterationResults))
                {
                    hasConverged = true;
                    break;
                }
            }

            vop.v_eq(x[i], x[i+1]);
            vop.v_eq(v[i], v[i+1]);

            msg_info() << iterationResults.str();
        }

        if (!hasConverged)
        {
            msg_warning() << "Newton's method failed to converge after " << newtonIterationCount
                << " iteration(s) with residual squared norm = " << squaredResidualNorm << ". ";

            if (!hasLineSearchFailed)
            {
                static constexpr auto divergedMaxIterations = NewtonStatus("DivergedMaxIterations");
                d_status.setValue(divergedMaxIterations);
            }

            if (d_updateStateWhenDiverged.getValue())
            {
                vop.v_eq(xResult, x[i+1]);
                vop.v_eq(vResult, v[i+1]);
            }
        }
    }
}

}
