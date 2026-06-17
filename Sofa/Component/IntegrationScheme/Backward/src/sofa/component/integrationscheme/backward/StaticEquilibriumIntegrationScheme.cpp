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
{  }

void StaticEquilibriumIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    simulation::common::VectorOperations::realloc(*m_vop, m_unknown, "dx", this, true);
    simulation::common::VectorOperations::realloc(*m_vop, m_r0, "r0", this, true);
}

/**
 * Compute the system matrix.
 */
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

/**
* compute the current RHS.
*/
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


/**
 * Returns the evaluation of the residue
 */
SReal StaticEquilibriumIntegrationScheme::evaluateResidue()
{
    core::behavior::MultiVecDeriv r0(m_vop.get(), m_r0);

    return r0.dot(r0);
}


/**
 * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
 */
void StaticEquilibriumIntegrationScheme::solveLinearEquation()
{
    SCOPED_TIMER("MBKSolve");

    l_linearSolver->getLinearSystem()->setSystemSolution(m_unknown);
    l_linearSolver->getLinearSystem()->setRHS(m_r0);
    l_linearSolver->solveSystem();
    l_linearSolver->getLinearSystem()->dispatchSystemSolution(m_unknown);
}

/**
 * Once (x^{i+1}-x^i) has been computed, the result is used internally to update the current
 * guess. It computes x^{i+1} += alpha * dx, where dx is the result of the linear system. It is
 * not necessary to share the result with the Newton-Raphson method.
 */
void StaticEquilibriumIntegrationScheme::updateStatesFromLinearSolution(SReal alpha, bool firstIteration)
{
    sofa::core::behavior::MultiVecCoord pos(m_vop.get(), m_xResult);

    pos.peq(m_unknown, alpha );
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

    //Compute current residual, usefull for static solver to return fast
    computeRHS(true);
    SReal oldResidue = evaluateResidue();
    SReal newResidue = evaluateResidue();


    unsigned it = 0;
    while ( it<maxNewtonIt && newResidue>residueThreshold )
    {
        const bool firstIt = it==0;

        if ( ! firstIt )
        {
            computeRHS(firstIt);
            oldResidue = evaluateResidue();
        }
        computeLHS(firstIt);
        //Find decrease direction
        solveLinearEquation();

        //Setup variables for linesearch
        SReal alpha = newtonStepSize;
        SReal delta = 0.0;

        //Already make a full step
        updateStatesFromLinearSolution(alpha, firstIt);
        computeRHS(false);
        newResidue = evaluateResidue();

        //Compute the Armijo term
        m_vop->v_dot(m_unknown, m_r0);
        const SReal armijoTerm = lineSearchArmijoFactor * m_vop->finish();

        unsigned lineSearchIt = 0;
        while ((newResidue>(oldResidue + alpha*armijoTerm)) && lineSearchIt<maxLineSearchIt )
        {
            //We are backtracking on the same line. Instead of starting from initial position and
            //adding alpha each time, we go back toward the initial position
            delta = alpha*lineSearchReductionRate;
            alpha -= delta;

            updateStatesFromLinearSolution(-delta, false);
            computeRHS(false);
            newResidue = evaluateResidue();

            ++lineSearchIt;
        }

        if (newResidue>oldResidue)
        {
            msg_warning()<<"Newton step increased the residual";
        }

        if (printLog)
        {
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

    d_currentResidue.setValue(newResidue);


}

void registerStaticEquilibriumIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator using implicit backward Euler scheme.")
        .add< StaticEquilibriumIntegrationScheme >());
}

} // namespace sofa::component::integrationscheme::forward
