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

#include <sofa/core/behavior/IntegrationScheme.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/behavior/MultiVec.h>

#include <sofa/core/behavior/LinearSolverAccessor.h>

namespace sofa::simulation::common
{
class MechanicalOperations;
class VectorOperations;
}

namespace sofa::simulation::integrationschemes
{

class SOFA_SIMULATION_CORE_API ImplicitIntegrationScheme :
                            public  sofa::core::behavior::IntegrationScheme,
                            public sofa::core::behavior::LinearSolverAccessor
{
public:
    SOFA_ABSTRACT_CLASS(ImplicitIntegrationScheme, sofa::core::behavior::IntegrationScheme);

    // WARNING we expect the linear integrator to initialize the working vecs. Meaning that if we
    // work in FreeMotion, the xResult should already be equal to the actual position.
    // Same for the velocity. This is expected when updating the position, only a += will be done.
    virtual void setupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);
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
     * Returns the evaluation of the residue
     */
    virtual SReal evaluateResidue() = 0;


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

    virtual void integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) final ;


    /// Given the solution dx of the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual SReal getVelocityIntegrationFactor() const = 0;

    /// Given the solution dx of the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual SReal getPositionIntegrationFactor() const = 0;

protected:

    virtual sofa::Size getIntegrationSchemeTimeOrder() const = 0;

    Data<SReal> d_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0


    const core::ExecParams* m_params;
    sofa::core::MultiVecCoordId m_xResult;
    sofa::core::MultiVecDerivId m_vResult;
    sofa::core::MultiVecDerivId m_unknown;

    sofa::core::MultiVecDerivId m_r0, m_r1, m_r2;

    std::vector<sofa::core::MultiVecCoordId> m_x0;
    std::vector<sofa::core::MultiVecDerivId> m_a0, m_v0;

    sofa::core::MultiVecDerivId m_acceleration;


    std::shared_ptr<sofa::simulation::common::VectorOperations > m_vop;
    std::unique_ptr<sofa::simulation::common::MechanicalOperations> m_mop;

};
} // namespace sofa::component::integrationschemes


