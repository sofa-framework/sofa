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
#include <sofa/simulation/Visitor.h>


namespace sofa::simulation
{

/**
 * Compute forces for interaction force fields which are not under an ODE solver
 *
 * Example: two objects are simulated each with its own dedicated ODE solver. It is possible to
 * connect both objects using a spring. In that case, the spring component can be defined
 * out of the objects Nodes. Warning: although this configuration is possible, it
 * is not recommended. Prefer to use a common ODE solver for both objects. Indeed,
 * those forces will not contribute to the global matrix system in the case
 * of an implicit time integration scheme.
 */
class SOFA_SIMULATION_CORE_API ComputeIsolatedForceVisitor : public Visitor
{
public:
    ComputeIsolatedForceVisitor(const core::ExecParams* execParams, const SReal dt);

    Result processNodeTopDown(simulation::Node* node) override;

protected:
    SReal m_dt{};

    void fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* obj);
};

}
