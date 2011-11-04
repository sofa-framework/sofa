/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/simulation/common/UpdateMappingVisitor.h>
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
    /*
        if (!obj->isMechanical())
        {
          simulation::Node::ctime_t t0=begin(n, obj);

    #ifdef SOFA_DUMP_VISITOR_INFO
      if (printActivated)      printNode("propagateX");
    #endif
            obj->propagateX();
    #ifdef SOFA_DUMP_VISITOR_INFO
      if (printActivated)      printCloseNode("propagateX");
    #endif


    #ifdef SOFA_DUMP_VISITOR_INFO
      if (printActivated)      printNode("propagateV");
    #endif
            obj->propagateV();
    #ifdef SOFA_DUMP_VISITOR_INFO
      if (printActivated)      printCloseNode("propagateV");
    #endif

          end(n, obj, t0);
        }
    */
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

