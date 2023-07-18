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

#include <sofa/simulation/StateChangeVisitor.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/topology/TopologicalMapping.h>
#include <sofa/core/BaseMapping.h>
#include <sofa/core/behavior/BaseMechanicalState.h>

namespace sofa
{

namespace simulation
{

StateChangeVisitor::StateChangeVisitor(const sofa::core::ExecParams* params, sofa::core::topology::Topology* source)
    : Visitor(params), root(true), m_source(source)
{
}

void StateChangeVisitor::processStateChange(sofa::core::behavior::BaseMechanicalState* obj)
{
    obj->handleStateChange(m_source);
}

Visitor::Result StateChangeVisitor::processNodeTopDown(simulation::Node* node)
{
    if (!root && node->mechanicalMapping)
    {
        if (!node->mechanicalMapping->sameTopology())
        {
            // stop all topological computations
            return RESULT_PRUNE;
        }
    }

    if (node->mechanicalState && testTags(node->mechanicalState))
    {
        this->processStateChange(node->mechanicalState);
    }

    // TODO 2019-01-04: epernod remove this hack when mechanicalMapping could be updated directly form MechanicalObject through Data link
    // search for mechanical mapping,
    for (simulation::Node::ObjectIterator it = node->object.begin(); it != node->object.end(); ++it)
    {
        sofa::core::BaseMapping* obj = dynamic_cast<sofa::core::BaseMapping*>(it->get());
        if (obj != nullptr)
        {
            const ctime_t t0=begin(node,obj);
            obj->handleTopologyChange(); // update the specific TopologicalMapping
            end(node,obj,t0);
        }
    }

    root = false; // now we process child nodes
    return RESULT_CONTINUE; // continue the propagation of state changes
}

} // namespace simulation

} // namespace sofa

