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
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/core/VecId.h>
#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/simulation/Node.h>
#include <sofa/core/BaseMapping.h>

namespace sofa
{

namespace simulation
{

void UpdateMappingVisitor::processMapping(simulation::Node* /*n*/, core::BaseMapping* obj)
{
    const std::string msg = "MappingVisitor - processMapping: " + obj->getName();
    sofa::helper::ScopedAdvancedTimer timer(msg.c_str());

    obj->apply(core::mechanicalparams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
    obj->applyJ(core::mechanicalparams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
}

void UpdateMappingVisitor::processMechanicalMapping(simulation::Node* /*n*/, core::BaseMapping* obj)
{
    // mechanical mappings with isMechanical flag not set are now processed by the MechanicalPropagatePositionVisitor visitor
    const std::string msg = "MappingVisitor - processMechanicalMapping: " + obj->getName();
    sofa::helper::ScopedAdvancedTimer timer(msg.c_str());
}

Visitor::Result UpdateMappingVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->mapping, &UpdateMappingVisitor::processMapping);
    for_each(this, node, node->mechanicalMapping, &UpdateMappingVisitor::processMechanicalMapping);

    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

