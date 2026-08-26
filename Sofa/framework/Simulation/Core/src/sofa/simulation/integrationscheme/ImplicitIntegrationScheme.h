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

#include <sofa/core/behavior/BaseIntegrationScheme.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MultiVec.h>

#include <sofa/core/behavior/LinearSolverAccessor.h>
#include <sofa/simulation/MappingGraphMechanicalOperations.h>
#include <sofa/simulation/MappingGraph.h>

namespace sofa::simulation::common
{
class MechanicalOperations;
class VectorOperations;
}

namespace sofa::simulation::integrationscheme
{

class SOFA_SIMULATION_CORE_API ImplicitIntegrationScheme :
                            public sofa::core::behavior::BaseIntegrationScheme,
                            public sofa::core::behavior::LinearSolverAccessor
{
public:
    SOFA_ABSTRACT_CLASS2(ImplicitIntegrationScheme, sofa::core::behavior::BaseIntegrationScheme, sofa::core::behavior::LinearSolverAccessor);

    /**
     * Template method pattern is implemented here to ensure some data/member are rightly initialized
     * before solving. This method internally calls doSetupIntegrationStep that can be overridden.
     */
    virtual void setupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) final;
    virtual void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
    {  }

    ImplicitIntegrationScheme();

    /**
     * Compute the system matrix.
     */
    virtual void computeLHS(bool firstIteration = false) = 0;

     /**
     * compute the current RHS.
     */
    virtual void computeRHS(bool firstIteration = false) = 0;


    /**
     * Returns the evaluation of the residual
     */
    virtual SReal evaluateResidual() = 0;


    /**
     * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
     */
    virtual void solveLinearEquation() = 0;

    /**
     * Once (x^{i+1}-x^i) has been computed, the result is used internally to update the current
     * guess. It computes x^{i+1} += alpha * dx, where dx is the result of the linear system. It is
     * not necessary to share the result with the Newton-Raphson method.
     */
    virtual void updateStatesFromLinearSolution(SReal alpha, bool firstIteration = false) = 0;

    /**
     * This method is called after the integration step is completed.
     */
    virtual void finalizeIntegrationStep()
    {  };

    /**
     * @param params Parameters for vector and mechanical operations
     * @param dt Time step for integration
     * @param xResult MultiVecCoordId in which to store the new position
     * @param vResult MultiVecCoordId in which to store the new velocity
     *
     * This method is a monolithic step integration. It is expected that given the dt, the position
     * and velocity in xResult and vResult are updated to their value at time t+dt
     */
    virtual void integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;


    /// The integration scheme solves a linear system. In this linear system the unknown has a unit.
    /// It can be the position, velocity or acceleration increase. In any case, this unknown need then
    /// to be integrated to update the velocity. This methods returns the factor to put in front of
    /// this unknown before accumulating it to the velocity.
    ///
    /// Said differently, if $x$ is the unknown, $v_{t}$ the velocity at time $t$, then we have
    /// $$
    /// v_{t+dt} = v_{t} + k*x + r
    /// $$
    /// with $r$ being constant in term of $x$ and $k$ the integration factor returned by this function.
    ///
    /// This method is used to compute the compliance for contact corrections.
    virtual SReal getVelocityIntegrationFactor() const = 0;

    /// The integration scheme solves a linear system. In this linear system the unknown has a unit.
    /// It can be the position, velocity or acceleration increase. In any case, this unknown need then
    /// to be integrated to update the velocity. This methods returns the factor to put in front of
    /// this unknown before accumulating it to the velocity.
    ///
    /// Said differently, if $x$ is the unknown, $p_{t}$ the position at time $t$, then we have
    /// $$
    /// p_{t+dt} = p_{t} + k*x + r
    /// $$
    /// with $r$ being constant in term of $x$ and $k$ the integration factor returned by this function.
    ///
    /// This method is used to compute the compliance for contact corrections.
    virtual SReal getPositionIntegrationFactor() const = 0;

    Data<SReal> d_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0


protected:

    /**
     * This method returns the order of the integration scheme in term of number of past timestep
     * needed to compute the next timestep.
     * For instance, if $p_{t+dt} = f(v_{t+dt}, ... , v_{t-k*dt}$, then the order is k+1
     */
    virtual sofa::Size getIntegrationSchemeTimeOrder() const = 0;

    const core::ExecParams* m_params;
    sofa::core::MultiVecCoordId m_xResult;
    sofa::core::MultiVecDerivId m_vResult;
    sofa::core::MultiVecDerivId m_systemUnknown;

    sofa::core::MultiVecDerivId m_r0, m_r1, m_r2;

    std::vector<sofa::core::MultiVecCoordId> m_x0;
    std::vector<sofa::core::MultiVecDerivId> m_a0, m_v0;
    bool m_passedStatesValid;

    sofa::core::MultiVecDerivId m_acceleration;


    std::shared_ptr<sofa::simulation::common::VectorOperations > m_vop;
    std::unique_ptr<sofa::simulation::common::MappingGraphMechanicalOperations> m_mop;

    sofa::simulation::MappingGraph m_mappingGraph;
};
} // namespace sofa::component::integrationscheme


