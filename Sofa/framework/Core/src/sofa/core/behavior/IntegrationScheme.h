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

#include <sofa/core/behavior/MultiVec.h>

namespace sofa::core::behavior
{

/**
 *  \brief Component responsible for timestep integration, i.e. advancing the state from time t to t+dt.
 *
 *  This class currently control both the integration scheme (explicit,
 *  implicit, static, etc).
 *
 *  While all computations required to do the integration step are handled by
 *  this object, they should not be implemented directly in it, but instead
 *  the solver propagates orders (or Visitor) to the other components in the
 *  scenegraph that will locally execute them. This allows for greater
 *  flexibility (the solver can just ask for the forces to be computed without
 *  knowing what type of forces are present), as well as performances
 *  (some computations can be executed in parallel).
 *
 */
class SOFA_CORE_API IntegrationScheme : public virtual objectmodel::BaseComponent
{
public:
    SOFA_ABSTRACT_CLASS(IntegrationScheme, objectmodel::BaseComponent);
    SOFA_BASE_CAST_IMPLEMENTATION(IntegrationScheme)

protected:
    IntegrationScheme();
    ~IntegrationScheme() override;

public:

    // WARNING we expect the linear integrator to initialize the working vecs. Meaning that if we
    // work in FreeMotion, the xResult should already be equal to the actual position.
    // Same for the velocity. This is expected when updating the position, only a += will be done.
    virtual void setupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult);
    virtual void doSetupIntegrationStep(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult)
    {  }

    /**
     * Compute the system matrix.
     */
    virtual void computeLHS(unsigned iteration = 0) = 0;

     /**
     * compute the current RHS.
     */
    virtual void computeRHS(unsigned iteration = 0) = 0;


    /**
     * Returns the squared norm of the last evaluation of the RHS
     */
    virtual SReal squaredNormRHS() = 0;


    /**
     * Solve the linear equation from a Newton iteration, i.e. it computes (x^{i+1}-x^i).
     */
    virtual void solveLinearEquation() = 0;

    /**
     * Once (x^{i+1}-x^i) has been computed, the result is used internally to update the current
     * guess. It computes x^{i+1} += alpha * dx, where dx is the result of the linear system. It is
     * not necessary to share the result with the Newton-Raphson method.
     */
    virtual void updateVelocityAndPositionFromLinearSolution(SReal alpha, unsigned iteration = 0) = 0;


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

protected:

    virtual sofa::Size getIntegrationSchemeOrder() = 0;

    const core::ExecParams* m_params;
    SReal m_dt;
    sofa::core::MultiVecCoordId m_xResult;
    sofa::core::MultiVecDerivId m_vResult;

    sofa::core::MultiVecDerivId m_r0, m_r1, m_r2;

    std::vector<MultiVecCoordId> m_x0;
    std::vector<MultiVecDerivId> m_a0, m_v0;

    sofa::core::MultiVecDerivId m_acceleration;

    sofa::core::MultiVecDerivId m_unknown;


    Data<SReal> d_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0


};

} // namespace sofa::core::behavior
