﻿/******************************************************************************
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
        core::MatricesFactors::M{1},
        core::MatricesFactors::B{-dt},
        core::MatricesFactors::K{-dt*dt}
    };
}

void BDF1::computeRightHandSide(
    const core::ExecParams* params,
    core::behavior::RHSInput input,
    core::MultiVecDerivId rightHandSide,
    SReal dt)
{
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(true);

    {
        SCOPED_TIMER("ComputeForce");
        static constexpr bool clearForcesBeforeComputingThem = true;
        static constexpr bool applyBottomUpMappings = true;

        mop.mparams.setX(input.intermediatePosition);
        mop.mparams.setV(input.intermediateVelocity);
        mop.computeForce(input.force,
            clearForcesBeforeComputingThem, applyBottomUpMappings);
    }

    // b = dt * f
    m_vop->v_eq(rightHandSide, input.force, dt);

    // b += M * (v_n - v_i)
    {
        core::behavior::MultiVecDeriv tmp(m_vop.get());
        tmp.peq(m_oldVelocity);
        m_vop->v_peq(tmp, input.intermediateVelocity, -1); //(v_n - v_i)
        mop.addMdx(rightHandSide, tmp, core::MatricesFactors::M(1).get());
    }

    // b += (dt^2 K) * v
    {
        const auto backupV = mop.mparams.v();
        {
            mop.mparams.setV(m_oldVelocity);
            mop.addMBKv(rightHandSide,
                core::MatricesFactors::M(0),
                core::MatricesFactors::B(0),
                core::MatricesFactors::K(dt * dt));
        }
        mop.mparams.setV(backupV);
    }

    // Apply projective constraints
    mop.projectResponse(rightHandSide);

}

void BDF1::updateStates(const core::ExecParams* params, SReal dt,
    core::MultiVecCoordId x, core::MultiVecDerivId v,
    core::MultiVecCoordId newX, core::MultiVecDerivId newV,
    core::MultiVecDerivId linearSystemSolution)
{
    // v_(i+1) = v_i + x
    m_vop->v_eq(newV, v);
    m_vop->v_peq(newV, linearSystemSolution);

    // x_(i+1) = dt * v_(i+1) + x_i
    m_vop->v_eq(newX, x);
    m_vop->v_peq(newX, newV, dt);
}

}