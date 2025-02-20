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
#include <sofa/component/odesolver/integration/BDF1.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>


namespace sofa::component::odesolver::integration
{

void registerBDF1(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Velocity-based Backward Euler integration method.")
        .add< BDF1 >());
}


BDF1::BDF1()
    : d_rayleighStiffness(initData(&d_rayleighStiffness, 0_sreal, "rayleighStiffness", "Rayleigh damping coefficient related to stiffness, > 0") )
    , d_rayleighMass(initData(&d_rayleighMass, 0_sreal, "rayleighMass", "Rayleigh damping coefficient related to mass, > 0"))
{}

void BDF1::initializeVectors(const core::ExecParams* params, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v)
{
    m_vop = std::make_unique<simulation::common::VectorOperations>(params, this->getContext());

    m_oldPosition.realloc(m_vop.get(), false, true, core::VecIdProperties{"oldPosition", GetClass()->className});
    m_oldVelocity.realloc(m_vop.get(), false, true, core::VecIdProperties{"oldVelocity", GetClass()->className});

    m_vop->v_eq(m_oldPosition, x);
    m_vop->v_eq(m_oldVelocity, v);
}

core::behavior::BaseIntegrationMethod::Factors BDF1::getMatricesFactors(SReal dt) const
{
    return {
        core::MatricesFactors::M{1 + d_rayleighMass.getValue() * dt},
        core::MatricesFactors::B{-dt},
        core::MatricesFactors::K{-dt * (d_rayleighStiffness.getValue() + dt)}
    };
}

void BDF1::computeRightHandSide(
    const core::ExecParams* params,
    core::behavior::RHSInput input,
    core::MultiVecDerivId force,
    core::MultiVecDerivId rightHandSide,
    SReal dt)
{
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(true);

    const auto alpha = d_rayleighMass.getValue();
    const auto beta = d_rayleighStiffness.getValue();

    const auto& x = input.intermediatePosition;
    const auto& v = input.intermediateVelocity;

    {
        SCOPED_TIMER("ComputeForce");
        static constexpr bool clearForcesBeforeComputingThem = true;
        static constexpr bool applyBottomUpMappings = true;

        mop.mparams.setX(x);
        mop.mparams.setV(v);
        mop.computeForce(force,
            clearForcesBeforeComputingThem, applyBottomUpMappings);
    }

    // b = dt * f
    m_vop->v_eq(rightHandSide, force, dt);

    // b += M * (v_n - (1 + dt * alpha) v_i)
    {
        core::behavior::MultiVecDeriv tmp(m_vop.get());
        tmp.eq(m_oldVelocity); // v_n

        //(v_n - (1 + dt * alpha) v_i)
        m_vop->v_peq(tmp, v, -(1 + dt * alpha));

        mop.addMdx(rightHandSide, tmp, core::MatricesFactors::M(1).get());
    }

    // b += dt * K * ((beta + dt) * v^i + x_n - x^i)
    {
        const auto backupV = mop.mparams.v();
        {
            core::behavior::MultiVecDeriv tmp(m_vop.get());
            
            tmp.eq(v, (beta + dt)); // (beta + dt) * v^i
            tmp.peq(m_oldPosition); // (beta + dt) * v^i + x_n
            tmp.peq(x, -1); // (beta + dt) * v^i + x_n - x^i
            
            mop.mparams.setV(tmp);
            mop.addMBKv(rightHandSide,
                core::MatricesFactors::M(0),
                core::MatricesFactors::B(0),
                core::MatricesFactors::K(dt));
        }
        mop.mparams.setV(backupV);
    }


    // Apply projective constraints
    mop.projectResponse(rightHandSide);

}

void BDF1::updateStates(const core::ExecParams* params, SReal dt, core::MultiVecCoordId x,
                        core::MultiVecDerivId v, core::MultiVecCoordId newX,
                        core::MultiVecDerivId newV, core::MultiVecDerivId linearSystemSolution)
{
    // v_(i+1) = v_i + x
    m_vop->v_eq(newV, v);
    m_vop->v_peq(newV, linearSystemSolution);

    // x_(i+1) = dt * v_(i+1) + x_n
    m_vop->v_eq(newX, x);
    m_vop->v_peq(newX, newV, dt);
}
SReal BDF1::computeResidual(const core::ExecParams* params, SReal dt, core::MultiVecDerivId force,
                            core::MultiVecDerivId oldVelocity, core::MultiVecDerivId newVelocity)
{
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    simulation::common::VectorOperations vop(params, this->getContext());

    core::behavior::MultiVecDeriv residual(
        &vop, true, core::VecIdProperties{"residual", GetClass()->className});

    // r = M (v - v_n)
    {
        core::behavior::MultiVecDeriv tmp(&vop);

        vop.v_eq(tmp, newVelocity);
        vop.v_peq(tmp, oldVelocity, -1);
        mop.addMdx(residual, tmp);
    }

    // r = -dt * F
    vop.v_peq(residual, force, -dt);

    // Apply projective constraints
    mop.projectResponse(residual);

    vop.v_dot(residual, residual);
    // msg_info() << residual;
    return vop.finish();
}

}  // namespace sofa::component::odesolver::integration
