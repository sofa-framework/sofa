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
#include <sofa/component/integrationscheme/backward/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/simulation/integrationscheme/ImplicitIntegrationScheme.h>

namespace sofa::simulation::common
{
class MechanicalOperations;
class VectorOperations;
}

namespace sofa::component::integrationscheme::backward
{

class SOFA_COMPONENT_INTEGRATIONSCHEME_BACKWARD_API StaticEquilibriumIntegrationScheme :
                            public sofa::simulation::integrationscheme::ImplicitIntegrationScheme
{
public:
    SOFA_CLASS(StaticEquilibriumIntegrationScheme, ImplicitIntegrationScheme);

    StaticEquilibriumIntegrationScheme();

    virtual void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    virtual void computeLHS(bool firstIteration = true) override;
    virtual void computeRHS(bool firstIteration = true) override;
    virtual SReal evaluateResidual() override;
    virtual void solveLinearEquation() override;
    virtual void updateStatesFromLinearSolution(SReal alpha, bool firstIteration = true) override;

    virtual SReal getVelocityIntegrationFactor() const override final;
    virtual SReal getPositionIntegrationFactor() const override final;

    virtual void integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    Data<unsigned int> d_maxNbIterationsNewton;
    Data<unsigned int> d_maxNbIterationsLineSearch;
    Data<SReal> d_newtonStepSize;
    Data<SReal> d_lineSearchReductionRate;
    Data<SReal> d_lineSearchArmijoFactor;
    Data<SReal> d_residueThreshold;
    Data<SReal> d_currentResidue;
    Data<bool> d_alwaysAdvanceNewton;

protected:

    virtual sofa::Size getIntegrationSchemeTimeOrder() const override;

};

} // namespace sofa::component::integrationscheme::backward
