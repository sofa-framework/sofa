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
 *  \brief Component responsible for linearizing the ODE by relating the current timestep to the next one linearly.
 *
 *  This is the base class of every type of integration scheme (explicit, implicit, static, etc).
 *
 *  Its API is pretty simple as the explicit family of solver wouldn't gain anything being
 *  solved by non-linear algorithms, so we only expect the classes to 'integrate' in a monolithic way
 *  The modularized versions will be implemented in the ImplicitIntegrationScheme API
 *
 */
class SOFA_CORE_API BaseIntegrationScheme : public virtual objectmodel::BaseComponent
{
public:
    SOFA_ABSTRACT_CLASS(BaseIntegrationScheme, objectmodel::BaseComponent);
    SOFA_BASE_CAST_IMPLEMENTATION(BaseIntegrationScheme)

protected:
    BaseIntegrationScheme();
    ~BaseIntegrationScheme() override;

public:

    /**
     * @param params Parameters for vector and mechanical operations
     * @param dt Time step for integration
     * @param xResult MultiVecCoordId in which to store the new position
     * @param vResult MultiVecCoordId in which to store the new velocity
     *
     * This method is a monolithic step integration. It is expected that given the dt, the position
     * and velocity in xResult and vResult are updated to their value at time t+dt
     */
    virtual void integrate(const core::ExecParams* params, SReal dt, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) = 0;

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


    bool insertInNode( objectmodel::BaseNode* node ) override;
    bool removeInNode( objectmodel::BaseNode* node ) override;

protected:

    SReal m_dt;
};

} // namespace sofa::core::behavior
