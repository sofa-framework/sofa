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
#include <sofa/simulation/integrationscheme/ImplicitIntegrationScheme.h>

#include <sofa/simulation/MechanicalOperations.h>
#include <sofa/simulation/VectorOperations.h>

namespace sofa::simulation::integrationscheme
{

/**
 * This class is a specialization of implicit integration scheme, where the unknown is expressed as
 * a delta in acceleration.
 *
 * Fixing this will then result in a particular expression of the gradient/hessian computation
 * as well as the state update, enabling to have a generic formulation of the ODE linearization
 * for all integration scheme that uses acceleration as the unknown.
 *
 * For more information see documentation : https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/
 */
class SOFA_SIMULATION_CORE_API AccelerationBasedImplicitIntegrationScheme :
                            public ImplicitIntegrationScheme
{
public:
    SOFA_ABSTRACT_CLASS(AccelerationBasedImplicitIntegrationScheme, ImplicitIntegrationScheme);

    AccelerationBasedImplicitIntegrationScheme() = default;

    /** Inherited for ImplicitIntegrationScheme **/
    /**
     *  All of those overriding derive from the equations presented in the documentation https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/#solving-for-non-linearities
     **/

    virtual void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    virtual void computeLHS(bool firstIteration = false) override;
    virtual void computeRHS(bool firstIteration = false) override;
    virtual SReal evaluateResidual() override;
    virtual void solveLinearEquation() override;
    virtual void updateStatesFromLinearSolution(SReal alpha, bool firstIteration = false) override;

    virtual SReal getVelocityIntegrationFactor() const override final;
    virtual SReal getPositionIntegrationFactor() const override final;

protected:
    virtual sofa::Size getIntegrationSchemeTimeOrder() const = 0;
    /** ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ **/

    /**   New API methods of AccelerationBasedImplicitIntegrationScheme   **/
    /**
     * This method returns a scalar which is the value of the derivative of the position integration
     * scheme with respect to the acceleration.
     *
     * In other words, if the position integration scheme is given by  $ p_{t+dt} = g_p(v_{t+dt}, a_{t+dt}) $
     * then this function returns $\derivative{g_p(v_{t+dt}, a_{t+dt})}{a_{t+dt}}$
     *
     * For more information see the documentation : https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/#the-sofa-implementation
     **/
    virtual SReal getPositionUpdateDerivedFromAcceleration() const = 0;

    /**
     * This method returns a scalar which is the value of the derivative of the position integration
     * scheme with respect to the velocity.
     *
     * In other words, if the position integration scheme is given by  $ p_{t+dt} = g_p(v_{t+dt}, a_{t+dt}) $
     * then this function returns $\derivative{g_p(v_{t+dt}, a_{t+dt})}{v_{t+dt}}$
     *
     * For more information see the documentation : https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/#the-sofa-implementation
     **/
    virtual SReal getPositionUpdateDerivedFromVelocity() const = 0;

    /**
     * This method returns a scalar which is the value of the derivative of the velocity integration
     * scheme with respect to the acceleration.
     *
     * In other words, if the position integration scheme is given by  $ v_{t+dt} = g_v(a_{t+dt}) $
     * then this function returns $\derivative{g_v(a_{t+dt})}{a_{t+dt}}$
     *
     * For more information see the documentation : https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/#the-sofa-implementation
     **/
    virtual SReal getVelocityUpdateDerivedFromAcceleration() const = 0;

    /**
     * This method compute the error in term of position update given the current position (in the
     * vecId position) and the current velocity (vecId velocity) and the current acceleration (vecId
     * acceleration) and store it into the VecId result.
     *
     * In equations the computation looks like this : $r = x_{t+h} - g_x(v,a)$
     *
     * For more information see the documentation : https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/#the-sofa-implementation
     **/
    virtual void computeCurrentPositionIntegrationError(sofa::simulation::common::VectorOperations & vop, sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecCoordId& position, const sofa::core::MultiVecDerivId& velocity, const sofa::core::MultiVecDerivId& acceleration) = 0;

    /**
     * This method compute the error in term of velocity update given the current velocity (vecId
     * velocity) and the current acceleration (vecId acceleration) and store it into the VecId result.
     *
     * In equations the computation looks like this : $r = v_{t+h} - g_v(a)$
     *
     * For more information see the documentation : https://sofa-framework.github.io/doc/simulation-principles/system-resolution/integration-scheme/#the-sofa-implementation
     **/
    virtual void computeCurrentVelocityIntegrationError(sofa::simulation::common::VectorOperations & vop, const sofa::core::MultiVecDerivId& result, const sofa::core::MultiVecDerivId& velocity, const sofa::core::MultiVecDerivId& acceleration) = 0;
};

} // namespace sofa::component::integrationscheme
