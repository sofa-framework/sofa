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
#include <sofa/component/odesolver/backward/Static.h>
#include <sofa/core/ObjectFactory.h>

#include "sofa/helper/ScopedAdvancedTimer.h"
#include "sofa/simulation/MechanicalOperations.h"
#include "sofa/simulation/VectorOperations.h"

namespace sofa::component::odesolver::integration
{

void registerStatic(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(
        core::ObjectRegistrationData("Static integration method.").add<Static>());
}

std::size_t Static::stepSize() const { return 1; }
void Static::initializeVectors(const core::ExecParams* params, core::ConstMultiVecCoordId x,
                               core::ConstMultiVecDerivId v)
{
    
}

core::behavior::BaseIntegrationMethod::Factors Static::getMatricesFactors(SReal dt) const
{
    return {
        core::MatricesFactors::M{0},
        core::MatricesFactors::B{0},
        core::MatricesFactors::K{1}
    };
}

void Static::computeRightHandSide(const core::ExecParams* params, core::behavior::RHSInput input,
                                  core::MultiVecDerivId force, core::MultiVecDerivId rightHandSide,
                                  SReal dt)
{
    x_i = input.intermediatePosition;
    
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    simulation::common::VectorOperations vop(params, this->getContext());
    
    mop->setImplicit(true);

    {
        SCOPED_TIMER("ComputeForce");
        static constexpr bool clearForcesBeforeComputingThem = true;
        static constexpr bool applyBottomUpMappings = true;

        mop.mparams.setX(x_i);
        mop.mparams.setV(input.intermediateVelocity);
        mop.computeForce(force,
            clearForcesBeforeComputingThem, applyBottomUpMappings);
    }

    // b = f
    vop.v_eq(rightHandSide, force, -1);

    mop.projectResponse(rightHandSide);
}

void Static::updateStates(const core::ExecParams* params, SReal dt, core::MultiVecCoordId x,
                          core::MultiVecDerivId v, core::MultiVecCoordId newX,
                          core::MultiVecDerivId newV, core::MultiVecDerivId linearSystemSolution)
{
    simulation::common::VectorOperations vop(params, this->getContext());
    vop.v_op(newX, x_i, linearSystemSolution, 1);
}

SReal Static::computeResidual(const core::ExecParams* params, SReal dt, core::MultiVecDerivId force,
                              core::MultiVecDerivId oldVelocity, core::MultiVecDerivId newVelocity)
{
    sofa::simulation::common::MechanicalOperations mop( params, this->getContext() );
    simulation::common::VectorOperations vop(params, this->getContext());

    core::behavior::MultiVecDeriv residual(
        &vop, true, core::VecIdProperties{"residual", GetClass()->className});

    // r = F
    vop.v_peq(residual, force, 1);

    // Apply projective constraints
    mop.projectResponse(residual);

    vop.v_dot(residual, residual);
    return vop.finish();
}

}  // namespace sofa::component::odesolver::integration