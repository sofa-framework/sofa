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
#include <sofa/simulation/config.h>

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MultiVec.h>
#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/simulation/integrationschemes/ImplicitIntegrationScheme.h>

namespace sofa::simulation::common
{
class MechanicalOperations;
class VectorOperations;
}

namespace sofa::component::integrationschemes::backward
{

class SOFA_SIMULATION_CORE_API StaticEquilibriumIntegrationScheme :
                            public sofa::simulation::integrationschemes::ImplicitIntegrationScheme
{
public:
    SOFA_ABSTRACT_CLASS(StaticEquilibriumIntegrationScheme, ImplicitIntegrationScheme);

    StaticEquilibriumIntegrationScheme() = default;

    void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    /**
     * Compute the system matrix.
     */
    void computeLHS(bool firstIteration = true) override;

    /**
    * compute the current RHS.
    */
    void computeRHS(bool firstIteration = true) override;

    /**
     * Returns the squared norm of the last evaluation of the RHS
     */
    SReal squaredNormRHS() override;

    /**
     * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
     */
    void solveLinearEquation() override;

    /**
     * Once (x^{i+1}-x^i) has been computed, the result is used internally to update the current
     * guess. It computes x^{i+1} += alpha * dx, where dx is the result of the linear system. It is
     * not necessary to share the result with the Newton-Raphson method.
     */
    void updateStatesFromLinearSolution(SReal alpha, bool firstIteration = true) override;

    virtual SReal getVelocityIntegrationFactor() const final;
    virtual SReal getPositionIntegrationFactor() const final;

protected:

    virtual sofa::Size getIntegrationSchemeTimeOrder() const;

};

} // namespace sofa::component::integrationschemes
