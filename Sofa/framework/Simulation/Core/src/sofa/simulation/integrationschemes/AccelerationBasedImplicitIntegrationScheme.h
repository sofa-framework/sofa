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

#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

namespace sofa::simulation::integrationschemes
{

class SOFA_SIMULATION_CORE_API AccelerationBasedImplicitIntegrationScheme :
                            public ImplicitIntegrationScheme
{
public:
    SOFA_ABSTRACT_CLASS(AccelerationBasedImplicitIntegrationScheme, ImplicitIntegrationScheme);

    AccelerationBasedImplicitIntegrationScheme() = default;

    void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    /**
     * Compute the system matrix.
     */
    void computeLHS(bool firstIteration = false) override;

    /**
    * compute the current RHS.
    */
    void computeRHS(bool firstIteration = false) override;


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
    void updateStatesFromLinearSolution(SReal alpha, bool firstIteration = false) override;

    virtual SReal getVelocityIntegrationFactor() const final;
    virtual SReal getPositionIntegrationFactor() const final;

protected:

    virtual sofa::Size getIntegrationSchemeOrder() const = 0;


    virtual SReal getPositionUpdateDerivedFromAcceleration() const = 0;
    virtual SReal getPositionUpdateDerivedFromVelocity() const = 0;
    virtual SReal getVelocityUpdateDerivedFromAcceleration() const = 0;

    //Compute the error made on the position integration equation : x_{t+h} - g_x(v,a), with v and a the current estimates of velocity and acceleration
    virtual void computeCurrentPositionIntegrationError(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity, const sofa::core::MultiVecDerivId& acceleration) = 0;
    //Compute the error made on the position integration equation : v_{t+h} - g_v(a), with a the current estimate of acceleration
    virtual void computeCurrentVelocityIntegrationError(sofa::simulation::common::VectorOperations & vop, const sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& acceleration) = 0;
};

} // namespace sofa::component::integrationschemes
