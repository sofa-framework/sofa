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
#pragma once
#include <sofa/component/odesolver/backward/config.h>
#include <sofa/core/behavior/BaseIntegrationMethod.h>

#include "sofa/core/behavior/MultiVec.h"
#include "sofa/simulation/VectorOperations.h"

namespace sofa::component::odesolver::integration
{
class SOFA_COMPONENT_ODESOLVER_BACKWARD_API Static : public sofa::core::behavior::BaseIntegrationMethod
{
public:
    SOFA_CLASS(Static, sofa::core::behavior::BaseIntegrationMethod);

    std::size_t stepSize() const override;
    void initializeVectors(const core::ExecParams* params, core::ConstMultiVecCoordId x, core::ConstMultiVecDerivId v) override;
    Factors getMatricesFactors(SReal dt) const override;
    void computeRightHandSide(const core::ExecParams* params, core::behavior::RHSInput input,
                              core::MultiVecDerivId force, core::MultiVecDerivId rightHandSide,
                              SReal dt) override;
    void updateStates(const core::ExecParams* params, SReal dt, core::MultiVecCoordId x,
                      core::MultiVecDerivId v, core::MultiVecCoordId newX,
                      core::MultiVecDerivId newV,
                      core::MultiVecDerivId linearSystemSolution) override;
    SReal computeResidual(const core::ExecParams* params,
        SReal dt,
        core::MultiVecDerivId force,
        core::MultiVecDerivId oldVelocity,
        core::MultiVecDerivId newVelocity) override;

private:
    core::MultiVecCoordId x_i; //x[i]
};
}