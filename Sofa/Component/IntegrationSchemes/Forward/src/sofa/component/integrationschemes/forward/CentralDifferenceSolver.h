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
#include <sofa/component/integrationschemes/forward/config.h>

#include <sofa/simulation/integrationschemes/ExplicitIntegrationScheme.h>

namespace sofa::component::integrationschemes::forward
{

/** Explicit time integrator using central difference (also known as Verlet of Leap-frop).
 *
 * @see http://www.dynasupport.com/support/tutorial/users.guide/time.integration
 * @see http://en.wikipedia.org/wiki/Leapfrog_method
 *
 */
class SOFA_COMPONENT_INTEGRATIONSCHEMES_FORWARD_API CentralDifferenceSolver : public simulation::integrationschemes::ExplicitIntegrationScheme
{
public:
    SOFA_CLASS(CentralDifferenceSolver, simulation::integrationschemes::ExplicitIntegrationScheme);
protected:
    CentralDifferenceSolver();
public:

    Data<SReal> d_rayleighMass; ///< Rayleigh damping coefficient related to mass
    Data<bool> d_threadSafeVisitor; ///< If true, do not use realloc and free visitors in fwdInteractionForceField.

    virtual void doIntegrate(const core::ExecParams* params, sofa::core::MultiVecCoordId xResult, sofa::core::MultiVecDerivId vResult) override;

    virtual SReal getVelocityIntegrationFactor() const override
    {
        return m_dt;
    }

    virtual SReal getPositionIntegrationFactor() const override
    {
        return m_dt * m_dt;
    }


};

} // namespace sofa::component::odesolver::forward
