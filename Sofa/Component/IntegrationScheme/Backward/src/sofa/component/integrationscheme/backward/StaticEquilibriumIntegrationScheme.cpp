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
#include <sofa/simulation/DifferentialOperations.h>
#include <sofa/simulation/MappingGraph.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalGetNonDiagonalMassesCountVisitor.h>

using sofa::simulation::mechanicalvisitor::MechanicalGetNonDiagonalMassesCountVisitor;


namespace sofa::component::integrationscheme::backward
{

StaticEquilibriumIntegrationScheme::StaticEquilibriumIntegrationScheme()
{  }

void StaticEquilibriumIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{
    SOFA_UNUSED(dt);
    SOFA_UNUSED(params);
    SOFA_UNUSED(vResult);
    SOFA_UNUSED(xResult);
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
    SOFA_UNUSED(firstIteration);
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
    SOFA_UNUSED(firstIteration);
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

void registerStaticEquilibriumIntegrationScheme(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Time integrator finding static equilibrium.")
        .add< StaticEquilibriumIntegrationScheme >());
}

} // namespace sofa::component::integrationscheme::backward
