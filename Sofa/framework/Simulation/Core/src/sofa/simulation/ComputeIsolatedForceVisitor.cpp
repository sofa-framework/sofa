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
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/MultiVecId.h>
#include <sofa/core/behavior/BaseInteractionForceField.h>
#include <sofa/simulation/ComputeIsolatedForceVisitor.h>
#include <sofa/simulation/Node.h>


namespace sofa::simulation
{

ComputeIsolatedForceVisitor::ComputeIsolatedForceVisitor(const core::ExecParams* execParams, const SReal dt)
    : Visitor(execParams)
    , m_dt(dt)
{}

Visitor::Result ComputeIsolatedForceVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!node->solver.empty() )
    {
        return RESULT_PRUNE;
    }
    for_each(this, node, node->interactionForceField, &ComputeIsolatedForceVisitor::fwdInteractionForceField);
    return RESULT_CONTINUE;
}

void ComputeIsolatedForceVisitor::fwdInteractionForceField(simulation::Node* node, core::behavior::BaseInteractionForceField* obj)
{
    SOFA_UNUSED(node);

    const core::MultiVecDerivId ffId = core::VecDerivId::externalForce();
    core::MechanicalParams mparams;
    mparams.setDt(m_dt);
    obj->addForce(&mparams, ffId);
}

}
