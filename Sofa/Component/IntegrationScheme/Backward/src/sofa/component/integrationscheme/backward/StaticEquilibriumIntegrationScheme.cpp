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
#include <sofa/component/integrationscheme/backward/StaticEquilibriumIntegrationScheme.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MappingGraph.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalGetNonDiagonalMassesCountVisitor.h>

using sofa::simulation::mechanicalvisitor::MechanicalGetNonDiagonalMassesCountVisitor;


namespace sofa::component::integrationscheme::backward
{

StaticEquilibriumIntegrationScheme::StaticEquilibriumIntegrationScheme()
: d_maxNbIterationsNewton(initData(&d_maxNbIterationsNewton, static_cast<unsigned int>(10) , "maxNbIterationsNewton", "Maximum number of iteration for the Newton algorithm"))
, d_maxNbIterationsLineSearch(initData(&d_maxNbIterationsLineSearch, static_cast<unsigned int>(1), "maxNbIterationsLineSearch", "Maximum number of iteration for the backtracking linesearch algorithm"))
, d_newtonStepSize(initData(&d_newtonStepSize, 1.0_sreal , "newtonStepSize", "Size of the first newton step before the linesearch"))
, d_lineSearchReductionRate(initData(&d_lineSearchReductionRate, 0.5_sreal , "lineSearchReductionRate", "Taken in [0,1[ representing the fraction of diminution of the step done in the backtracking line search (if set to 0.3, the first line search will reduce the step from 1.0 to 0.7)"))
, d_lineSearchArmijoFactor(initData(&d_lineSearchArmijoFactor, 1e-3_sreal , "lineSearchArmijoFactor", "Taken in [0,1[ it represents a tolerance on the residue in term of the linear approximation. e.g., for a value of 0.01, it means we want the solution to decrease the residue as much as 0.01 times the linear approximation in the same direction."))
, d_residueThreshold(initData(&d_residueThreshold, 1e-9_sreal , "residueThreshold", "Threshold under which, the residue is considered to be sufficiently low. Newton algorithm will stop after reaching a lower value"))
, d_currentResidue(initData(&d_currentResidue , "currentResidue", "Current value of the residue"))
, d_alwaysAdvanceNewton(initData(&d_alwaysAdvanceNewton , false, "alwaysAdvanceNewton", "Even if the linesearch didn't find a better solution than the current one, take the best one along the path that is not the current guess."))
{  }

void StaticEquilibriumIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    simulation::common::VectorOperations::realloc(*m_vop, m_systemUnknown, "dx", this, true);
    simulation::common::VectorOperations::realloc(*m_vop, m_r0, "r0", this, true);
}

void StaticEquilibriumIntegrationScheme::computeLHS(bool firstIteration)
{
    SOFA_UNUSED(firstIteration);

    {

        SCOPED_TIMER("setSystemMBKMatrix");
        const core::MatricesFactors::M mFact( 0 );
        const core::MatricesFactors::B bFact( 0 );
        const core::MatricesFactors::K kFact( -1.0 );

        m_mop->setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());
    }

}

void StaticEquilibriumIntegrationScheme::computeRHS(bool firstIteration)
{
    sofa::core::behavior::MultiVecDeriv f(m_vop.get(), core::vec_id::write_access::force );
    f.clear();

    {
        //TODO deal with that.
        SCOPED_TIMER("ComputeForce");
        m_mop->mparams.setImplicit(true); // this solver is implicit
        // compute the net forces at the beginning of the time step
        m_mop->computeForce(m_mappingGraph, f, true, true, nullptr); //f = Kx + Bv

        m_mop->projectResponse(m_mappingGraph,f);   // b is projected to the constrained space

        m_vop->v_eq(m_r0,core::vec_id::write_access::force );
    }

}

SReal StaticEquilibriumIntegrationScheme::evaluateResidual()
{
    core::behavior::MultiVecDeriv r0(m_vop.get(), m_r0);

    return r0.dot(r0);
}

void StaticEquilibriumIntegrationScheme::solveLinearEquation()
{
    SCOPED_TIMER("MBKSolve");

    l_linearSolver->getLinearSystem()->setSystemSolution(m_systemUnknown);
    l_linearSolver->getLinearSystem()->setRHS(m_r0);
    l_linearSolver->solveSystem();
    l_linearSolver->getLinearSystem()->dispatchSystemSolution(m_systemUnknown);
}

void StaticEquilibriumIntegrationScheme::updateStatesFromLinearSolution(SReal alpha, bool firstIteration)
{
    sofa::core::behavior::MultiVecCoord pos(m_vop.get(), m_xResult);

    pos.peq(m_systemUnknown, alpha );
}

SReal StaticEquilibriumIntegrationScheme::getVelocityIntegrationFactor() const
{
    return 0.0;
}

SReal StaticEquilibriumIntegrationScheme::getPositionIntegrationFactor() const
{
    return 1.0;
}

sofa::Size  StaticEquilibriumIntegrationScheme::getIntegrationSchemeTimeOrder() const
{
    return 1;
}

void StaticEquilibriumIntegrationScheme::integrate(const core::ExecParams* params, SReal dt,
                                                   sofa::core::MultiVecCoordId xResult,
                                                   sofa::core::MultiVecDerivId vResult)
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

    //Setup tue integration step
    setupIntegrationStep(params, dt, xResult, vResult);

    //Compute current residual, useful for static solver to return fast
    computeRHS(true);
    SReal oldResidue = evaluateResidual();
    SReal newResidue = evaluateResidual();


    unsigned it = 0;
    while ( it<maxNewtonIt && newResidue>residueThreshold )
    {
        SCOPED_TIMER_VARNAME(step_timer, "NewtonStep");

        const bool firstIt = it==0;

        // If in first iteration, this has already been computed earlier
        if ( ! firstIt )
        {
            computeRHS(firstIt);
            oldResidue = evaluateResidual();
        }

        double bestresidual = oldResidue;
        double bestalpha = 0.0;

        computeLHS(firstIt);
        //Find decrease direction
        solveLinearEquation();

        //Setup variables for linesearch
        SReal alpha = newtonStepSize;
        SReal delta = 0.0;

        //Already make a full step
        updateStatesFromLinearSolution(alpha, firstIt);
        m_mop->propagateX(xResult); //Need to propagate explicitly to enable recomputation of mapped Forcefield

        computeRHS(false);
        newResidue = evaluateResidual();

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

            updateStatesFromLinearSolution(-delta, false);
            m_mop->propagateX(xResult); //Need to propagate explicitly to enable recomputation of mapped Forcefield

            computeRHS(false);
            newResidue = evaluateResidual();

            if (newResidue<bestresidual)
            {
                bestresidual = newResidue;
                bestalpha = alpha;
            }

            ++lineSearchIt;
        }

        if (fabs(alpha - bestalpha )> std::numeric_limits<SReal>::epsilon() )
        {
            updateStatesFromLinearSolution( bestalpha - alpha, false);
            m_mop->propagateX(xResult); //Need to propagate explicitly to enable recomputation of mapped Forcefileld

            computeRHS(false);
            newResidue = evaluateResidual();
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

void registerStaticEquilibriumIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator finding static equilibrium.")
        .add< StaticEquilibriumIntegrationScheme >());
}

} // namespace sofa::component::integrationscheme::backward
