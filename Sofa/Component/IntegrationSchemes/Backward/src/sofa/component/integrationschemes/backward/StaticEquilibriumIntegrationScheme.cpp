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
#include <sofa/component/integrationschemes/backward/StaticEquilibriumIntegrationScheme.h>
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


namespace sofa::component::integrationschemes::backward
{
void StaticEquilibriumIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );

    simulation::common::VectorOperations::realloc(vop, m_unknown, "dx", this, true);
}

/**
 * Compute the system matrix.
 */
void StaticEquilibriumIntegrationScheme::computeLHS(bool firstIteration)
{
    SOFA_UNUSED(firstIteration);

    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );

    {

        SCOPED_TIMER("setSystemMBKMatrix");
        const core::MatricesFactors::M mFact( 0 );
        const core::MatricesFactors::B bFact( 0 );
        const core::MatricesFactors::K kFact( -1.0 );

        mop.setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());
    }

}

/**
* compute the current RHS.
*/
void StaticEquilibriumIntegrationScheme::computeRHS(bool firstIteration)
{
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );
    sofa::core::behavior::MultiVecDeriv f(&vop, core::vec_id::write_access::force );
    f.clear();

    {
        //TODO deal with that.
        SCOPED_TIMER("ComputeForce");
        mop->setImplicit(true); // this solver is implicit
        // compute the net forces at the beginning of the time step
        mop.computeForce(f);                                                               //f = Kx + Bv

        mop.projectResponse(f);   // b is projected to the constrained space
    }

}


/**
 * Returns the squared norm of the last evaluation of the RHS
 */
SReal StaticEquilibriumIntegrationScheme::squaredNormRHS()
{
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );

    core::behavior::MultiVecDeriv r0(&vop, core::vec_id::write_access::force);

    return r0.dot(r0);
}


/**
 * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
 */
void StaticEquilibriumIntegrationScheme::solveLinearEquation()
{
    SCOPED_TIMER("MBKSolve");

    l_linearSolver->getLinearSystem()->setSystemSolution(m_unknown);
    l_linearSolver->getLinearSystem()->setRHS(core::vec_id::write_access::force);
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
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );

    sofa::core::behavior::MultiVecCoord pos(&vop, m_xResult);

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


void registerStaticEquilibriumIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator using implicit backward Euler scheme.")
        .add< StaticEquilibriumIntegrationScheme >());
}

} // namespace sofa::component::integrationschemes::forward
