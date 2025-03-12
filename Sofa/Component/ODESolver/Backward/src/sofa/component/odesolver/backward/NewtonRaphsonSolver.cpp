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
#include <sofa/component/odesolver/backward/convergence/AbsoluteConvergenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/AbsoluteEstimateDifferenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/RelativeEstimateDifferenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/RelativeInitialConvergenceMeasure.h>
#include <sofa/component/odesolver/backward/convergence/RelativeSuccessiveConvergenceMeasure.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa::component::odesolver::backward
{

void registerNewtonRaphsonSolver(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Generic Newton-Raphson algorithm solving nonlinear equations.")
        .add< NewtonRaphsonSolver >());
}

static constexpr NewtonStatus defaultStatus("Undefined");

NewtonRaphsonSolver::NewtonRaphsonSolver()
    : d_maxNbIterationsNewton(
          initData(&d_maxNbIterationsNewton, 1u, "maxNbIterationsNewton",
                   "Maximum number of iterations of the Newton's method if it has not converged."))
    , d_relativeSuccessiveStoppingThreshold(initData(
          &d_relativeSuccessiveStoppingThreshold, 1e-5_sreal, "relativeSuccessiveStoppingThreshold",
          "Threshold for the relative successive progress criterion. The Newton "
          "iterations will stop when the ratio between the norm of the residual "
          "at iteration k over the norm of the residual at iteration k-1 is lower"
          " than this threshold."))
    , d_relativeInitialStoppingThreshold(initData(
          &d_relativeInitialStoppingThreshold, 1e-5_sreal, "relativeInitialStoppingThreshold",
          "Threshold for the relative initial progress criterion. The Newton"
          " iterations will stop when the ratio between the norm of the residual "
          "at iteration k over the norm of the residual at iteration 0 is"
          " lower than this threshold. This criterion tracks the overall progress "
          "made since the beginning of the iteration process. If the ratio is "
          "significantly smaller than 1, it indicates that the iterative process "
          "is making substantial progress, and the method is converging toward the"
          " root."))
    , d_absoluteResidualStoppingThreshold(initData(
          &d_absoluteResidualStoppingThreshold, 1e-5_sreal, "absoluteResidualStoppingThreshold",
          "Threshold for the absolute function value stopping criterion. The "
          "Newton iterations will stop when the norm of the residual at iteration "
          "k is lower than this threshold. This criterion indicates the current "
          "iteration found an estimate close to the root."))
    , d_relativeEstimateDifferenceThreshold(initData(
          &d_relativeEstimateDifferenceThreshold, 0_sreal, "relativeEstimateDifferenceThreshold",
          "Threshold for the relative change in root estimate criterion. The "
          "Newton iterations will stop when the difference between two successive "
          "estimates divided by the previous estimate is smaller than this threshold"))
    , d_absoluteEstimateDifferenceThreshold(initData(
          &d_absoluteEstimateDifferenceThreshold, 0_sreal, "absoluteEstimateDifferenceThreshold",
          "Threshold for the absolute change in root estimate criterion. The "
          "Newton iterations will stop when the difference between two successive "
          "estimates is smaller than this threshold."))
    , d_maxNbIterationsLineSearch(initData(
          &d_maxNbIterationsLineSearch, 5u, "maxNbIterationsLineSearch",
          "Maximum number of iterations of the line search method if it has not converged."))
    , d_lineSearchCoefficient(initData(&d_lineSearchCoefficient, 0.5_sreal, "lineSearchCoefficient",
                                       "Line search coefficient"))
    , d_updateStateWhenDiverged(initData(&d_updateStateWhenDiverged, true,
                                         "updateStateWhenDiverged",
                                         "Update the states within the last iteration even if the "
                                         "iterative process is considered diverged."))
    , d_status(initData(&d_status, defaultStatus, "status",
                        ("status\n" + NewtonStatus::dataDescription()).c_str()))
    , d_residualGraph(
          initData(&d_residualGraph, "residualGraph", "Graph of the residual over the iterations"))
    , d_warnWhenLineSearchFails(initData(&d_warnWhenLineSearchFails, true, "warnWhenLineSearchFails", "Trigger a warning if line search fails"))
    , d_warnWhenDiverge(initData(&d_warnWhenDiverge, true, "warnWhenDiverge", "Trigger a warning if Newton-Raphson diverges"))
{
    d_status.setReadOnly(true);

    static std::string groupAnalysis{"Analysis"};
    d_status.setGroup(groupAnalysis);
    d_residualGraph.setGroup(groupAnalysis);

    static std::string groupLineSearch{"Line Search"};
    d_maxNbIterationsLineSearch.setGroup(groupLineSearch);
    d_lineSearchCoefficient.setGroup(groupLineSearch);

    static std::string groupStoppingCriteria{"Stopping criteria"};
    d_maxNbIterationsNewton.setGroup(groupStoppingCriteria);
    d_relativeSuccessiveStoppingThreshold.setGroup(groupStoppingCriteria);
    d_relativeInitialStoppingThreshold.setGroup(groupStoppingCriteria);
    d_absoluteResidualStoppingThreshold.setGroup(groupStoppingCriteria);
    d_relativeEstimateDifferenceThreshold.setGroup(groupStoppingCriteria);
    d_absoluteEstimateDifferenceThreshold.setGroup(groupStoppingCriteria);

    d_residualGraph.setWidget("graph");
}

void NewtonRaphsonSolver::init()
{
    Inherit1::init();

    if (this->d_componentState.getValue() != core::objectmodel::ComponentState::Invalid)
    {
        d_componentState.setValue(core::objectmodel::ComponentState::Valid);
    }
}
void NewtonRaphsonSolver::reset()
{
    d_status.setValue(defaultStatus);

    auto graph = sofa::helper::getWriteAccessor(d_residualGraph);
    graph->clear();
}

void NewtonRaphsonSolver::initialConvergence(SReal squaredResidualNorm,
                                             const SReal squaredAbsoluteStoppingThreshold)
{
    msg_info() << "The equation to solve is satisfied with the initial guess. "
               << "The residual squared norm is " << squaredResidualNorm << ". "
               << "The threshold for convergence is " << squaredAbsoluteStoppingThreshold;
    static constexpr auto convergedEquilibrium = NewtonStatus("ConvergedEquilibrium");
    d_status.setValue(convergedEquilibrium);
}

bool NewtonRaphsonSolver::measureConvergence(const NewtonRaphsonConvergenceMeasure& measure,
                                             std::stringstream& os)
{
    if (measure.isMeasured())
    {
        if (measure.hasConverged())
        {
            d_status.setValue(measure.status());

            msg_info() << os.str();
            msg_info() << "[CONVERGED] " << measure.writeWhenConverged();

            return true;
        }
        else
        {
            const auto print = measure.writeWhenNotConverged();
            if (!print.empty())
            {
                os << "\n* " << print;
            }
        }
    }
    else if (notMuted())
    {
        os << "\n* " << measure.measureName() << ": NOT TESTED";
    }
    return false;
}

void NewtonRaphsonSolver::lineSearchIteration(
    newton_raphson::BaseNonLinearFunction& function, SReal& squaredResidualNorm,
    SReal lineSearchAlpha)
{
    // compute x^{i+1} += alpha * dx
    function.updateGuessFromLinearSolution(lineSearchAlpha);

    // compute r(x^{i+1})
    function.evaluateCurrentGuess();

    // compute ||r(x^{i+1}||
    squaredResidualNorm = function.squaredNormLastEvaluation();
}

struct NewtonIterationRAII
{
    explicit NewtonIterationRAII(newton_raphson::BaseNonLinearFunction& function) : function(function)
    {
        function.startNewtonIteration();
    }

    ~NewtonIterationRAII()
    {
        function.endNewtonIteration();
    }

    newton_raphson::BaseNonLinearFunction& function;
};

void NewtonRaphsonSolver::solve(newton_raphson::BaseNonLinearFunction& function)
{
    if (!this->isComponentStateValid())
    {
        return;
    }

    start();

    const bool printLog = f_printLog.getValue();
    auto graphAccessor = sofa::helper::getWriteAccessor(d_residualGraph);
    auto& graph = graphAccessor.wref();

    auto& residualList = graph["residual"];
    residualList.clear();

    SReal squaredResidualNorm{};
    {
        SCOPED_TIMER("ComputeError");

        // compute r(x^i)
        function.evaluateCurrentGuess();

        // compute ||r(x^i)||
        squaredResidualNorm = function.squaredNormLastEvaluation();

        residualList.push_back(squaredResidualNorm);
    }

    const auto absoluteStoppingThreshold = d_absoluteResidualStoppingThreshold.getValue();
    const auto squaredAbsoluteStoppingThreshold = std::pow(absoluteStoppingThreshold, 2);

    if (absoluteStoppingThreshold > 0 && squaredResidualNorm <= squaredAbsoluteStoppingThreshold)
    {
        initialConvergence(squaredResidualNorm, squaredAbsoluteStoppingThreshold);
    }
    else
    {
        SCOPED_TIMER("NewtonsIterations");

        msg_info() << "Initial squared residual norm: " << squaredResidualNorm;

        const auto relativeSuccessiveStoppingThreshold = d_relativeSuccessiveStoppingThreshold.getValue();

        RelativeSuccessiveConvergenceMeasure relativeSuccessiveConvergenceMeasure(relativeSuccessiveStoppingThreshold);
        RelativeInitialConvergenceMeasure relativeInitialConvergenceMeasure(d_relativeInitialStoppingThreshold.getValue());
        relativeInitialConvergenceMeasure.firstSquaredResidualNorm = squaredResidualNorm;
        AbsoluteConvergenceMeasure absoluteConvergenceMeasure(absoluteStoppingThreshold);
        AbsoluteEstimateDifferenceMeasure absoluteEstimateDifferenceMeasure(d_absoluteEstimateDifferenceThreshold.getValue());
        RelativeEstimateDifferenceMeasure relativeEstimateDifferenceMeasure(d_relativeEstimateDifferenceThreshold.getValue());

        const auto maxNbIterationsNewton = d_maxNbIterationsNewton.getValue();
        const auto maxNbIterationsLineSearch = std::max(d_maxNbIterationsLineSearch.getValue(), 1u);
        bool hasConverged = false;
        const auto lineSearchCoefficient = d_lineSearchCoefficient.getValue();

        unsigned int newtonIterationCount = 0;
        for (; newtonIterationCount < maxNbIterationsNewton; ++newtonIterationCount)
        {
            NewtonIterationRAII newtonIteration(function);

            msg_info() << "Newton iteration #" << newtonIterationCount;

            const auto previousSquaredResidualNorm = squaredResidualNorm;

            relativeSuccessiveConvergenceMeasure.newtonIterationCount = newtonIterationCount;
            relativeInitialConvergenceMeasure.newtonIterationCount = newtonIterationCount;
            absoluteConvergenceMeasure.newtonIterationCount = newtonIterationCount;
            absoluteEstimateDifferenceMeasure.newtonIterationCount = newtonIterationCount;
            relativeEstimateDifferenceMeasure.newtonIterationCount = newtonIterationCount;

            if (relativeEstimateDifferenceMeasure.isMeasured())
            {
                relativeEstimateDifferenceMeasure.squaredPreviousEvaluation = function.squaredLastEvaluation();
            }

            // compute J_r(x^i)
            function.computeGradientFromCurrentGuess();

            // solve J_r(x^i) * dx == -r(x^i)
            function.solveLinearEquation();

            SReal lineSearchAlpha = 1_sreal;
            SReal previousAlpha = 0;

            bool lineSearchSuccess = false;
            unsigned int lineSearchIterationCount = 0;
            for (; lineSearchIterationCount < maxNbIterationsLineSearch; ++lineSearchIterationCount)
            {
                lineSearchIteration(function, squaredResidualNorm, lineSearchAlpha - previousAlpha);

                if (squaredResidualNorm < previousSquaredResidualNorm)
                {
                    lineSearchSuccess = true;
                    break;
                }

                previousAlpha = lineSearchAlpha;
                lineSearchAlpha *= lineSearchCoefficient;
            }

            if (!lineSearchSuccess)
            {
                msg_warning_when(d_warnWhenLineSearchFails.getValue())
                    << "Line search failed at Newton iteration "
                    << newtonIterationCount << ".";

                static constexpr auto divergedMaxIterations = NewtonStatus("DivergedLineSearch");
                d_status.setValue(divergedMaxIterations);

                if (maxNbIterationsLineSearch > 1)
                {
                    lineSearchAlpha = 1_sreal;
                    lineSearchIteration(function, squaredResidualNorm, lineSearchAlpha - previousAlpha);
                }
            }
            else
            {
                msg_info() << "Line search succeeded after " << (lineSearchIterationCount+1) << " iterations";
            }

            residualList.push_back(squaredResidualNorm);

            std::stringstream iterationResults;
            if (printLog)
            {
                iterationResults << "Newton iteration results:";
                iterationResults << "\n* Current iteration = " << newtonIterationCount;
                iterationResults << "\n* Squared residual norm = " << squaredResidualNorm;
                iterationResults << "\n* Line search status: " << (lineSearchSuccess ? "SUCCESSFUL" : "FAILED");
                iterationResults << "\n* Residual norm = " << std::sqrt(squaredResidualNorm) << " (absolute threshold = " << absoluteStoppingThreshold << ")";
                iterationResults << "\n* Successive relative ratio = " << std::sqrt(squaredResidualNorm / previousSquaredResidualNorm) << " (threshold = " << relativeSuccessiveStoppingThreshold << ", previous residual norm = " << std::sqrt(previousSquaredResidualNorm) << ")";
            }

            relativeSuccessiveConvergenceMeasure.squaredResidualNorm = squaredResidualNorm;
            relativeSuccessiveConvergenceMeasure.previousSquaredResidualNorm = previousSquaredResidualNorm;
            relativeSuccessiveConvergenceMeasure.newtonIterationCount = newtonIterationCount;

            if (measureConvergence(relativeSuccessiveConvergenceMeasure, iterationResults))
            {
                hasConverged = true;
                break;
            }

            relativeInitialConvergenceMeasure.squaredResidualNorm = squaredResidualNorm;
            if (measureConvergence(relativeInitialConvergenceMeasure, iterationResults))
            {
                hasConverged = true;
                break;
            }

            absoluteConvergenceMeasure.squaredResidualNorm = squaredResidualNorm;
            if (measureConvergence(absoluteConvergenceMeasure, iterationResults))
            {
                hasConverged = true;
                break;
            }

            if (absoluteEstimateDifferenceMeasure.isMeasured()
                || relativeEstimateDifferenceMeasure.isMeasured())
            {
                const auto squaredAbsoluteDifference = function.squaredNormDx();
                if (printLog)
                {
                    iterationResults << "\n* Successive estimate difference = " << std::sqrt(squaredAbsoluteDifference);
                }

                relativeEstimateDifferenceMeasure.squaredAbsoluteDifference = squaredAbsoluteDifference;
                if (measureConvergence(relativeEstimateDifferenceMeasure, iterationResults))
                {
                    hasConverged = true;
                    break;
                }

                absoluteEstimateDifferenceMeasure.squaredAbsoluteDifference = squaredAbsoluteDifference;
                if (measureConvergence(absoluteEstimateDifferenceMeasure, iterationResults))
                {
                    hasConverged = true;
                    break;
                }
            }

            msg_info() << iterationResults.str();
        }

        if (!hasConverged)
        {
            msg_warning_when(d_warnWhenDiverge.getValue())
                << "Newton-Raphson method failed to converge after " << newtonIterationCount
                << " iteration(s) with residual squared norm = " << squaredResidualNorm << ". ";

            static constexpr auto divergedMaxIterations = NewtonStatus("DivergedMaxIterations");
            d_status.setValue(divergedMaxIterations);
        }
    }
}

void NewtonRaphsonSolver::start()
{
    // The status of the algorithm is set to "Running", and will be changed later
    // depending on the convergence of the algorithm.
    static constexpr auto running = NewtonStatus("Running");
    d_status.setValue(running);
}

}  // namespace sofa::component::odesolver::backward
