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
#include <sofa/core/ConstraintParams.h>
#include <sofa/simulation/NewtonRaphson.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/IntegrateVisitor.h>
#include <sofa/simulation/events/AnimateBeginEvent.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/task/MainTaskSchedulerFactory.h>
#include <sofa/simulation/task/TaskScheduler.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalPropagateOnlyPositionAndVelocityVisitor.h>


namespace sofa::simulation
{
NewtonRaphson::NewtonRaphson()
: d_maxNbIterationsNewton(initData(&d_maxNbIterationsNewton, static_cast<unsigned int>(10) , "maxNbIterationsNewton", "Maximum number of iteration for the Newton algorithm"))
, d_maxNbIterationsLineSearch(initData(&d_maxNbIterationsLineSearch, static_cast<unsigned int>(1), "maxNbIterationsLineSearch", "Maximum number of iteration for the backtracking linesearch algorithm"))
, d_newtonStepSize(initData(&d_newtonStepSize, 1.0_sreal , "newtonStepSize", "Size of the first newton step before the linesearch"))
, d_lineSearchReductionRate(initData(&d_lineSearchReductionRate, 0.5_sreal , "lineSearchReductionRate", "Taken in [0,1[ representing the fraction of diminution of the step done in the backtracking line search (if set to 0.3, the first line search will reduce the step from 1.0 to 0.7)"))
, d_lineSearchArmijoFactor(initData(&d_lineSearchArmijoFactor, 1e-3_sreal , "lineSearchArmijoFactor", "Taken in [0,1[ it represents a tolerance on the residue in term of the linear approximation. e.g., for a value of 0.01, it means we want the solution to decrease the residue as much as 0.01 times the linear approximation in the same direction."))
, d_residueThreshold(initData(&d_residueThreshold, 1e-9_sreal , "residueThreshold", "Threshold under which, the residue is considered to be sufficiently low. Newton algorithm will stop after reaching a lower value"))
, d_currentResidue(initData(&d_currentResidue , "currentResidue", "Current value of the residue"))
, d_alwaysAdvanceNewton(initData(&d_alwaysAdvanceNewton , false, "alwaysAdvanceNewton", "Even if the linesearch didn't find a better solution than the current one, take the best one along the path that is not the current guess."))
{
}

NewtonRaphson::~NewtonRaphson() = default;

void NewtonRaphson::init()
{
    // Inherit::init();
    if (!l_node)
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);
    }
}

void NewtonRaphson::integrate(const core::ExecParams* params, SReal dt)
{
    auto xResult = sofa::core::vec_id::write_access::position;
    auto vResult = sofa::core::vec_id::write_access::velocity;

    newtonSolve(params,dt, xResult, vResult);
}

void NewtonRaphson::newtonSolve(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xId, sofa::core::MultiVecDerivId vId)
{
     SCOPED_TIMER("StaticEquilibriumIntegrationScheme::Integrate");


    //Constify the data values
    const unsigned maxNewtonIt = d_maxNbIterationsNewton.getValue();
    const unsigned maxLineSearchIt = d_maxNbIterationsLineSearch.getValue();
    const SReal newtonStepSize = d_newtonStepSize.getValue();
    const SReal residueThreshold = d_residueThreshold.getValue();
    const SReal lineSearchReductionRate = d_lineSearchReductionRate.getValue();
    const SReal lineSearchArmijoFactor = d_lineSearchArmijoFactor.getValue();

    const bool printLog = f_printLog.getValue();

    IntegrationHelper intHelper(params, dt,  xId, vId, true, true);
    auto* rootNode = static_cast<sofa::simulation::Node*>(this->l_node.get());
    intHelper.setNode(rootNode);

    auto mop = std::make_unique<sofa::simulation::common::MappingGraphMechanicalOperations >( params, this->getContext() );


    //Setup tue integration step
    intHelper.setupIntegrationStep();

    //Compute current residual, useful for static solver to return fast
    intHelper.computeRHS(true);
    SReal oldResidue = intHelper.evaluateResidual();
    SReal newResidue = oldResidue;


    unsigned it = 0;
    while ( it<maxNewtonIt && newResidue>residueThreshold )
    {
        SCOPED_TIMER_VARNAME(step_timer, "NewtonStep");

        const bool firstIt = it==0;

        // If in first iteration, this has already been computed earlier
        if ( ! firstIt )
        {
            intHelper.computeRHS(firstIt);
            oldResidue = intHelper.evaluateResidual();
        }

        double bestresidual = oldResidue;
        double bestalpha = 0.0;

        intHelper.computeLHS(firstIt);
        //Find decrease direction
        intHelper.solveLinearEquation();

        //Setup variables for linesearch
        SReal alpha = newtonStepSize;
        SReal delta = 0.0;

        //Already make a full step
        intHelper.updateStatesFromLinearSolution(alpha, firstIt);
        mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

        intHelper.computeRHS(false);
        newResidue = intHelper.evaluateResidual();

        // Compute the Armijo term
        // !! Careful this approximation holds only when A is symmetric
        // This comes from the fact that if the current residual is f=||r||**2 and the search
        // direction results from Ap = -r, then we have \nabla f = 2*A*r, so the armijo factor can be
        // simplified with \nabla f * p = 2*r^T*A^T*A^{-1}*(-r) = -2*||r||^2  if A is symmetric
        // Otherwise we need to compute 2 * r^{t} * A^{T} * p
        const SReal armijoTerm = - 2 * lineSearchArmijoFactor * oldResidue;

        if (newResidue<bestresidual || d_alwaysAdvanceNewton.getValue())
        {
            bestresidual = newResidue;
            bestalpha = alpha;
        }

        unsigned lineSearchIt = 0;
        while ((newResidue>(oldResidue + alpha*armijoTerm)) && lineSearchIt<maxLineSearchIt )
        {
            //We are backtracking on the same line. Instead of starting from initial position and
            //adding alpha each time, we go back toward the initial position
            delta = alpha*lineSearchReductionRate;
            alpha -= delta;

            intHelper.updateStatesFromLinearSolution(-delta, false);
            mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

            intHelper.computeRHS(false);
            newResidue = intHelper.evaluateResidual();

            if (newResidue<bestresidual)
            {
                bestresidual = newResidue;
                bestalpha = alpha;
            }

            ++lineSearchIt;
        }

        if (fabs(alpha - bestalpha )> std::numeric_limits<SReal>::epsilon() )
        {
            intHelper.updateStatesFromLinearSolution( bestalpha - alpha, false);
            mop->propagateXAndV(xId, vId); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

            intHelper.computeRHS(false);
            newResidue = intHelper.evaluateResidual();
        }


        if (printLog)
        {
            if (newResidue>oldResidue)
            {
                msg_warning()<<"Newton step increased the residual";
            }
            msg_info()<<"Newton step = "<<it;
            msg_info()<<"Current residue = "<<newResidue<< "   | previous residue = "<<oldResidue ;
            msg_info()<<"Number of line search iterations = "<<lineSearchIt;
        }
        ++it ;
    }
    if (printLog)
    {
        if (newResidue<residueThreshold)
        {
            msg_info()<<"Newton converged to residue "<<newResidue<<" in "<<it<<" steps.";
        }
        else
        {
            msg_warning()<<"Newton didn't converge ! Current residue is "<<newResidue;
        }
    }

    sofa::helper::AdvancedTimer::valSet("nb_iterations", it);
    sofa::helper::AdvancedTimer::valSet("residual", std::sqrt(newResidue));

    d_currentResidue.setValue(newResidue);

}

void registerNewtonRaphson(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("TODO")
        .add<NewtonRaphson>()
        .addDocumentationURL(std::string(sofa::SOFA_DOCUMENTATION_URL) + std::string("TODO"))
        .addDescription(R"(TODO)"));
}

} // namespace sofa
