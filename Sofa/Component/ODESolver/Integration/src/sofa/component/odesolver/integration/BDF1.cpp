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

void BDF1::initializeVectors(core::MultiVecCoordId x, core::MultiVecDerivId v)
{
    m_x = x;
    m_v = v;
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
    core::MultiVecDerivId forceId,
    core::MultiVecDerivId rightHandSide,
    SReal dt)
{
    sofa::simulation::common::VectorOperations vop( params, this->getContext() );
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    mop->setImplicit(true);

    {
        SCOPED_TIMER("ComputeForce");
        static constexpr bool clearForcesBeforeComputingThem = true;
        static constexpr bool applyBottomUpMappings = true;

        // compute the net forces at the beginning of the time step
        mop.computeForce(forceId,
            clearForcesBeforeComputingThem, applyBottomUpMappings);
        msg_info() << "initial force = " << forceId;
    }

    // b = dt * f
    vop.v_eq(rightHandSide, forceId, dt);

    // b += (M + dt^2 K) * v
    mop.mparams.setV(m_v);
    mop.addMBKv(rightHandSide,
        core::MatricesFactors::M(1),
        core::MatricesFactors::B(0),
        core::MatricesFactors::K(dt * dt));

}

}
