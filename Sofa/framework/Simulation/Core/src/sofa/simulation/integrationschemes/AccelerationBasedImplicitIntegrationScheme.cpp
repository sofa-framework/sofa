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
#include <sofa/simulation/MappingGraph.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>
#include <sofa/simulation/mechanicalvisitor/MechanicalGetNonDiagonalMassesCountVisitor.h>

using sofa::simulation::mechanicalvisitor::MechanicalGetNonDiagonalMassesCountVisitor;


namespace sofa::simulation::integrationschemes
{
void AccelerationBasedImplicitIntegrationScheme::doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
{

    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );

    const Size order = getIntegrationSchemeOrder();

    m_x0.resize(order);
    m_v0.resize(order);
    m_a0.resize(order);

    simulation::common::VectorOperations::realloc(vop, m_r0, "r0", this, true);
    simulation::common::VectorOperations::realloc(vop, m_r1, "r1", this, true);
    simulation::common::VectorOperations::realloc(vop, m_r2, "r2", this, true);

    for (unsigned i = 0; i < order; ++i)
    {
        simulation::common::VectorOperations::realloc(vop, m_x0[i], "x0" + (order != 1 ? "_" + std::to_string(i)  : ""), this);
        simulation::common::VectorOperations::realloc(vop, m_v0[i], "v0" + (order != 1 ? "_" + std::to_string(i)  : ""), this);
        simulation::common::VectorOperations::realloc(vop, m_a0[i], "a0" + (order != 1 ? "_" + std::to_string(i)  : ""), this, true);
        if (this->getTime() < std::numeric_limits<SReal>::epsilon())
        {
            sofa::core::behavior::MultiVecDeriv v0(&vop, m_v0[i]);
            v0.eq(core::vec_id::write_access::velocity);
            sofa::core::behavior::MultiVecCoord x0(&vop, m_x0[i]);
            x0.eq(core::vec_id::write_access::position);
            sofa::core::behavior::MultiVecDeriv a0(&vop, m_a0[i]);
            a0.clear();
        }
    }
    for (unsigned i = 0; i < order - 1; ++i)
    {
        sofa::core::behavior::MultiVecCoord x(&vop, m_x0[i]);
        x.eq(m_x0[i+1]);
        sofa::core::behavior::MultiVecDeriv v(&vop, m_v0[i]);
        v.eq(m_v0[i+1]);
        sofa::core::behavior::MultiVecDeriv a(&vop, m_a0[i]);
        a.eq(m_a0[i+1]);
    }

    simulation::common::VectorOperations::realloc(vop, m_acceleration, "acceleration", this, true);
    simulation::common::VectorOperations::realloc(vop, m_unknown, "da", this, true);

    //Might be used afterwards by computeAccelerationFromVelocity
    sofa::core::behavior::MultiVecDeriv v0(&vop, m_v0[order - 1]);
    v0.eq(core::vec_id::write_access::velocity);
    sofa::core::behavior::MultiVecCoord x0(&vop, m_x0[order - 1]);
    x0.eq(core::vec_id::write_access::position);
    sofa::core::behavior::MultiVecDeriv a0(&vop, m_a0[order - 1]);
    a0.eq(m_acceleration);

}

/**
 * Compute the system matrix.
 */
void AccelerationBasedImplicitIntegrationScheme::computeLHS(unsigned iteration)
{
    SOFA_UNUSED(iteration);

    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );

    {
        SReal DGx = getPositionUpdateDerivedFromVelocity()*getVelocityUpdateDerivedFromAcceleration() + getPositionUpdateDerivedFromAcceleration();
        SReal DGv = getVelocityUpdateDerivedFromAcceleration();

        SCOPED_TIMER("setSystemMBKMatrix");
        const core::MatricesFactors::M mFact( 1 + DGv * d_rayleighMass.getValue() );
        const core::MatricesFactors::B bFact( -DGv );
        const core::MatricesFactors::K kFact( - DGx - DGv * d_rayleighStiffness.getValue() );

        mop.setSystemMBKMatrix(mFact, bFact, kFact, l_linearSolver.get());
    }

}

/**
* compute the current RHS.
*/
void AccelerationBasedImplicitIntegrationScheme::computeRHS(unsigned iteration)
{
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );
    sofa::core::behavior::MultiVecCoord pos(&vop, core::vec_id::write_access::position );
    sofa::core::behavior::MultiVecDeriv vel(&vop, core::vec_id::write_access::velocity );

    sofa::core::behavior::MultiVecDeriv f(&vop, core::vec_id::write_access::force );
    f.clear();
    sofa::core::behavior::MultiVecDeriv b(&vop, m_r0 );
    b.clear();
    sofa::core::behavior::MultiVecDeriv r1(&vop, m_r1 );
    r1.clear();
    sofa::core::behavior::MultiVecDeriv r2(&vop, m_r2 );
    r2.clear();



    {
        //TODO deal with that.
        SCOPED_TIMER("ComputeForce");
        mop->setImplicit(true); // this solver is implicit
        // compute the net forces at the beginning of the time step
        mop.computeForce(f);                                                               //f = Kx + Bv
    }

    {
        SCOPED_TIMER("ComputeRHTerm");

        b.eq(f, 1.0);  // b = f

        if (   fabs(d_rayleighMass.getValue()) > std::numeric_limits<SReal>::epsilon()
            || fabs(d_rayleighMass.getValue()) > std::numeric_limits<SReal>::epsilon())
        {
            mop.addMBKv(b, core::MatricesFactors::M(-d_rayleighMass.getValue()),
            core::MatricesFactors::B(0),
            core::MatricesFactors::K(d_rayleighStiffness.getValue()));
        }


        if (iteration == 0) [[unlikely]]
        {
            auto backV = mop->v();

            computeCurrentPositionIntegrationError(vop, m_r1, core::vec_id::write_access::velocity, m_acceleration);

            mop->setV(m_r1);
            mop.addMBKv(b, core::MatricesFactors::M(0.0),
                        core::MatricesFactors::B(0),
                        core::MatricesFactors::K(-1.0));


            computeCurrentVelocityIntegrationError(vop, m_r2, m_acceleration);

            mop->setV(m_r2);
            mop.addMBKv(b, core::MatricesFactors::M(d_rayleighMass.getValue()),
                        core::MatricesFactors::B(-1.0),
                        core::MatricesFactors::K(-getPositionUpdateDerivedFromVelocity() - d_rayleighStiffness.getValue()));
            mop->setV(backV);
        }


        auto backV = mop->v();
        mop->setV(m_acceleration);
        mop.addMBKv(b, core::MatricesFactors::M(-1.0),
                    core::MatricesFactors::B(0),
                    core::MatricesFactors::K(0));
        mop->setV(backV);

        mop.projectResponse(b);   // b is projected to the constrained space
    }

}


/**
 * Returns the squared norm of the last evaluation of the RHS
 */
SReal AccelerationBasedImplicitIntegrationScheme::squaredNormRHS()
{
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );

    core::behavior::MultiVecDeriv r0(&vop, m_r0);
    core::behavior::MultiVecDeriv r1(&vop, m_r1);

    return r0.dot(r0) + r1.dot(r1);
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
void AccelerationBasedImplicitIntegrationScheme::updateStatesFromLinearSolution(SReal alpha, unsigned iteration)
{
    sofa::simulation::common::VectorOperations vop( m_params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( m_params, this->getContext() );

    sofa::core::behavior::MultiVecCoord pos(&vop, m_xResult);
    sofa::core::behavior::MultiVecDeriv vel(&vop, m_vResult );
    sofa::core::behavior::MultiVecDeriv acc(&vop, m_acceleration );

    const SReal DGx = getPositionUpdateDerivedFromVelocity()*getVelocityUpdateDerivedFromAcceleration() + getPositionUpdateDerivedFromAcceleration();
    const SReal DGv = getVelocityUpdateDerivedFromAcceleration();


    acc.peq(m_unknown, alpha);
    vel.peq(m_unknown, alpha * DGv);
    pos.peq(m_unknown, alpha * DGx);

    //TODO make this work with alpha, iteration might be still 0 but we are in the linesearch algo and we don't want to remove this each time...
    if (iteration == 0) [[unlikely]]
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
