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

    virtual void integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) = 0;


    /// Given the solution dx of the linear system inversion, how much will it affect the velocity
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual SReal getVelocityIntegrationFactor() const = 0;

    /// Given the solution dx of the linear system inversion, how much will it affect the position
    ///
    /// This method is used to compute the compliance for contact corrections
    virtual SReal getPositionIntegrationFactor() const = 0;


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

    Data<SReal> d_rayleighStiffness; ///< Rayleigh damping coefficient related to stiffness, > 0
    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass, > 0


protected:


    const core::ExecParams* m_params;
    SReal m_dt;
    sofa::core::MultiVecCoordId m_xResult;
    sofa::core::MultiVecDerivId m_vResult;
    sofa::core::MultiVecDerivId m_unknown;




};

} // namespace sofa::core::behavior
