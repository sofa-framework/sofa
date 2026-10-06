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

#include <sofa/core/objectmodel/BaseComponent.h>
#include <sofa/core/behavior/BaseAnimationLoop.h>

#include <sofa/simulation/fwd.h>


namespace sofa::core
{
class ExecParams;
}


namespace sofa::simulation
{

/**
 *  \brief Default Animation Loop to be created when no AnimationLoop found on simulation::node.
 *
 *
 */
class SOFA_SIMULATION_CORE_API NewtonRaphson : public sofa::core::behavior::BaseTimeIntegrator
{
public:
    typedef sofa::core::behavior::BaseAnimationLoop Inherit;
    typedef sofa::core::objectmodel::BaseContext BaseContext;
    SOFA_CLASS(NewtonRaphson, sofa::core::behavior::BaseTimeIntegrator);

    virtual void init() override;
    virtual void integrate(const core::ExecParams* params, SReal dt) override;

protected:
    NewtonRaphson();
    ~NewtonRaphson() override;

};

} // namespace sofa::simulation
