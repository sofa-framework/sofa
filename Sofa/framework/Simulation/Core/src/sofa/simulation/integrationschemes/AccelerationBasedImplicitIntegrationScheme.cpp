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
#include <sofa/simulation/integrationschemes/AccelerationBasedImplicitIntegrationScheme.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/BaseMass.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalGetNonDiagonalMassesCountVisitor.h>

using sofa::simulation::mechanicalvisitor::MechanicalGetNonDiagonalMassesCountVisitor;


namespace sofa::simulation::integrationschemes
{

void AccelerationBasedImplicitIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{

    // Make sure the states are following the sizes
    simulation::common::VectorOperations::realloc(*m_vop, m_r0, "r0", this, true);
    simulation::common::VectorOperations::realloc(*m_vop, m_r1, "r1", this, true);
    simulation::common::VectorOperations::realloc(*m_vop, m_r2, "r2", this, true);
    simulation::common::VectorOperations::realloc(*m_vop, m_acceleration, "acceleration", this, true);
    simulation::common::VectorOperations::realloc(*m_vop, m_unknown, "da", this, true);

    // Deal with higher order integration scheme
    const Size order = getIntegrationSchemeTimeOrder();

    m_x0.resize(order);
    m_v0.resize(order);
    m_a0.resize(order);

    for (unsigned i = 0; i < order; ++i)
    {
        // If first order, don't add the subscript
        simulation::common::VectorOperations::realloc(*m_vop, m_x0[i], "x0" + (order != 1 ? "_" + std::to_string(i)  : ""), this);
        simulation::common::VectorOperations::realloc(*m_vop, m_v0[i], "v0" + (order != 1 ? "_" + std::to_string(i)  : ""), this);
        simulation::common::VectorOperations::realloc(*m_vop, m_a0[i], "a0" + (order != 1 ? "_" + std::to_string(i)  : ""), this, true);

        // Only at the start of the simulation, copy the position/velocity values inside the stored
        // states to recreate the past
        if (this->getTime() < std::numeric_limits<SReal>::epsilon())
        {
            sofa::core::behavior::MultiVecDeriv v0(m_vop.get(), m_v0[i]);
            v0.eq(core::vec_id::write_access::velocity);
            sofa::core::behavior::MultiVecCoord x0(m_vop.get(), m_x0[i]);
            x0.eq(core::vec_id::write_access::position);
            sofa::core::behavior::MultiVecDeriv a0(m_vop.get(), m_a0[i]);
            a0.clear();
        }
    }

    // Now shift all states to advance in time (could be skipped at the start of the simulation)
    // I decided not to do it to avoid having the check
    for (unsigned i = 0; i < order - 1; ++i)
    {
        sofa::core::behavior::MultiVecCoord x(m_vop.get(), m_x0[i]);
        x.eq(m_x0[i+1]);
        sofa::core::behavior::MultiVecDeriv v(m_vop.get(), m_v0[i]);
        v.eq(m_v0[i+1]);
        sofa::core::behavior::MultiVecDeriv a(m_vop.get(), m_a0[i]);
        a.eq(m_a0[i+1]);
    }

    // Store the previous state in its right position in the state vector
    sofa::core::behavior::MultiVecDeriv v0(m_vop.get(), m_v0[order - 1]);
    v0.eq(core::vec_id::write_access::velocity);
    sofa::core::behavior::MultiVecCoord x0(m_vop.get(), m_x0[order - 1]);
    x0.eq(core::vec_id::write_access::position);
    sofa::core::behavior::MultiVecDeriv a0(m_vop.get(), m_a0[order - 1]);
    a0.eq(m_acceleration);

    // This is only there for lagrangian based simulation, to make sure we start using the real pose
    // instead of the free pos (same for velocity)
    m_vop->v_eq(m_vResult, core::vec_id::write_access::velocity);
    m_vop->v_eq(m_xResult, core::vec_id::write_access::position);

}

/**
 * Compute the system matrix.
 */
void AccelerationBasedImplicitIntegrationScheme::computeLHS(bool firstIteration)
{
    SOFA_UNUSED(firstIteration);

    // Those two factors relate the position (DGx) and the velocity (DGv) to the acceleration
    // We can think of them as the first order approximation of the IS in term of acceleration
    const SReal DGx = getPositionUpdateDerivedFromVelocity()*getVelocityUpdateDerivedFromAcceleration() + getPositionUpdateDerivedFromAcceleration();
    const SReal DGv = getVelocityUpdateDerivedFromAcceleration();

    // Set the factor of the left hand side taking into account the rayleigh damping
    SCOPED_TIMER("setSystemMBKMatrix");
    const core::MatricesFactors::M mFact( 1 + DGv * d_rayleighMass.getValue() );
    const core::MatricesFactors::B bFact( -DGv );
    const core::MatricesFactors::K kFact( - DGx - DGv * d_rayleighStiffness.getValue() );

    m_mop->setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());
}

/**
* compute the current RHS.
*/
void AccelerationBasedImplicitIntegrationScheme::computeRHS(bool firstIteration)
{

    // Make sure no one modified this
    m_mop->cparams.setX(m_xResult);
    m_mop->cparams.setV(m_vResult);


    sofa::core::behavior::MultiVecDeriv f(m_vop.get(), core::vec_id::write_access::force );
    // Let's make sure f is cleared between each Newton steps
    f.clear();
    // The other one don't need to be clear as they are recomputed and not updated by the real IS impl

    {
        SCOPED_TIMER("ComputeForce");
        m_mop->mparams.setImplicit(true); // this solver is implicit
        // compute the net forces at the beginning of the time step
        m_mop->computeForce(f);
    }

    {
        SCOPED_TIMER("ComputeRHTerm");

        m_vop->v_eq(m_r0, f, 1.0);

        // This computes the explicit part of the Rayleigh damping
        if (   fabs(d_rayleighMass.getValue()) > std::numeric_limits<SReal>::epsilon()
            || fabs(d_rayleighStiffness.getValue()) > std::numeric_limits<SReal>::epsilon())
        {
            m_mop->addMBKv(m_r0, core::MatricesFactors::M(-d_rayleighMass.getValue()),
            core::MatricesFactors::B(0),
            core::MatricesFactors::K(d_rayleighStiffness.getValue()));
        }



        auto backV = m_mop->mparams.v();

        // R1 and R2 should be equal to 0 in theory when integration scheme is linear. But let's
        // recompute them anyway to compute the residue
        computeCurrentPositionIntegrationError (*m_vop, m_r1, core::vec_id::write_access::velocity, m_acceleration);
        computeCurrentVelocityIntegrationError (*m_vop, m_r2, m_acceleration);

        if (firstIteration)
        {
            m_mop->mparams.setV(m_r1);
            m_mop->addMBKv(m_r0, core::MatricesFactors::M(0.0),
                           core::MatricesFactors::B(0),
                           core::MatricesFactors::K(-1.0));

            m_mop->mparams.setV(m_r2);
            m_mop->addMBKv(m_r0, core::MatricesFactors::M(d_rayleighMass.getValue()),
                        core::MatricesFactors::B(-1.0),
                        core::MatricesFactors::K(-getPositionUpdateDerivedFromVelocity() - d_rayleighStiffness.getValue()));
        }

        m_mop->mparams.setV(m_acceleration);
        m_mop->addMBKv(m_r0, core::MatricesFactors::M(-1.0),
                    core::MatricesFactors::B(0),
                    core::MatricesFactors::K(0));
        m_mop->mparams.setV(backV);

        // Apply projective constraints to the full residue
        m_mop->projectResponse(m_r0);
        m_mop->projectResponse(m_r1);
        m_mop->projectResponse(m_r2);
    }

}


/**
 * Returns the squared norm of the last evaluation of the RHS
 */
SReal AccelerationBasedImplicitIntegrationScheme::squaredNormRHS()
{
    core::behavior::MultiVecDeriv r0(m_vop.get(), m_r0);
    core::behavior::MultiVecDeriv r1(m_vop.get(), m_r1);
    core::behavior::MultiVecDeriv r2(m_vop.get(), m_r2);

    return r0.dot(r0) + r1.dot(r1) + r2.dot(r2);
}


/**
 * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
 */
void AccelerationBasedImplicitIntegrationScheme::solveLinearEquation()
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
void AccelerationBasedImplicitIntegrationScheme::updateStatesFromLinearSolution(SReal alpha, bool firstIteration)
{
    sofa::core::behavior::MultiVecCoord pos(m_vop.get(), m_xResult);
    sofa::core::behavior::MultiVecDeriv vel(m_vop.get(), m_vResult );
    sofa::core::behavior::MultiVecDeriv acc(m_vop.get(), m_acceleration );


    // Those two factors relate the position (DGx) and the velocity (DGv) to the acceleration
    // We can think of them as the first order approximation of the IS in term of acceleration
    const SReal DGx = getPositionUpdateDerivedFromVelocity()*getVelocityUpdateDerivedFromAcceleration() + getPositionUpdateDerivedFromAcceleration();
    const SReal DGv = getVelocityUpdateDerivedFromAcceleration();

    //Update position, velocity and acceleration w/r to unknown
    acc.peq(m_unknown, alpha);
    vel.peq(m_unknown, alpha * DGv);
    pos.peq(m_unknown, alpha * DGx);

    //TODO make this work with alpha, iteration might be still 0 but we are in the linesearch algo and we don't want to remove this each time...
    if (firstIteration)
    {
        vel.peq(m_r2, -1.0);
        pos.peq(m_r1, -1.0);

        const SReal pudfv = getPositionUpdateDerivedFromVelocity();
        if ( fabs(pudfv) > std::numeric_limits<SReal>::epsilon() )
            pos.peq(m_r2, -pudfv);
    }
}



SReal AccelerationBasedImplicitIntegrationScheme::getVelocityIntegrationFactor() const
{
    return getVelocityUpdateDerivedFromAcceleration();
}

SReal AccelerationBasedImplicitIntegrationScheme::getPositionIntegrationFactor() const
{
    //TODO not 100% sure this is what's expected.
    //TODO But given what's in LinearSolverConstraintCorrection< DataTypes >::applyMotionCorrection it seems like it
    return getPositionUpdateDerivedFromVelocity()*getVelocityUpdateDerivedFromAcceleration() + getPositionUpdateDerivedFromAcceleration();
}




} // namespace sofa::component::integrationschemes::forward
