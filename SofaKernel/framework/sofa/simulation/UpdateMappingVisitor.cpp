/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
//#include <sofa/component/System.h>

namespace sofa
{

namespace simulation
{

void UpdateMappingVisitor::processMapping(simulation::Node* /*n*/, core::BaseMapping* obj)
{
    obj->apply(core::MechanicalParams::defaultInstance(), core::VecCoordId::position(), core::ConstVecCoordId::position());
    obj->applyJ(core::MechanicalParams::defaultInstance(), core::VecDerivId::velocity(), core::ConstVecDerivId::velocity());
}

void UpdateMappingVisitor::processMechanicalMapping(simulation::Node* /*n*/, core::BaseMapping* /*obj*/)
{
    // mechanical mappings with isMechanical flag not set are now processed by the MechanicalPropagatePositionVisitor visitor
}

Visitor::Result UpdateMappingVisitor::processNodeTopDown(simulation::Node* node)
{
    for_each(this, node, node->mapping, &UpdateMappingVisitor::processMapping);
    for_each(this, node, node->mechanicalMapping, &UpdateMappingVisitor::processMechanicalMapping);

//    {
//            if (!node->nodeInVisualGraph.empty()) node->nodeInVisualGraph->execute<UpdateMappingVisitor>();
//            for (simulation::Node::ChildIterator itChild = node->childInVisualGraph.begin(); itChild != node->childInVisualGraph.end(); ++itChild)
//            {
//                simulation::Node *child=*itChild;
//                child->execute<UpdateMappingVisitor>();
//            }
//    }

    return RESULT_CONTINUE;
}

} // namespace simulation

} // namespace sofa

