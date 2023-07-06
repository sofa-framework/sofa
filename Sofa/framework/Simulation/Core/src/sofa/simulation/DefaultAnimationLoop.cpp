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
#include <sofa/simulation/DefaultAnimationLoop.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/simulation/Node.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>

#include <sofa/helper/ScopedAdvancedTimer.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>

namespace sofa
{

namespace simulation
{

int DefaultAnimationLoopClass = core::RegisterObject("Simulation loop to use in scene without constraints nor contact.")
        .add< DefaultAnimationLoop >()
        .addDescription(R"(
This loop do the following steps:
- build and solve all linear systems in the scene : collision and time integration to compute the new values of the dofs
- update the context (dt++)
- update the mappings
- update the bounding box (volume covering all objects of the scene))");


DefaultAnimationLoop::~DefaultAnimationLoop()
{

}

void DefaultAnimationLoop::setNode( simulation::Node* n )
{
    l_node.set(n);
}

void DefaultAnimationLoop::step(const core::ExecParams* params, SReal dt)
{
    simulation::Node* node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    if (dt == 0)
        dt = node->getDt();

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    const SReal startTime = gnode->getTime();

    sofa::helper::AdvancedTimer::stepBegin("BehaviorUpdatePositionVisitor");
    BehaviorUpdatePositionVisitor beh(params , dt);
    node->execute ( beh );
    sofa::helper::AdvancedTimer::stepEnd("BehaviorUpdatePositionVisitor");


    sofa::helper::AdvancedTimer::stepBegin("UpdateInternalDataVisitor");
    UpdateInternalDataVisitor uid(params);
    node->execute ( uid );
    sofa::helper::AdvancedTimer::stepEnd("UpdateInternalDataVisitor");


    sofa::helper::AdvancedTimer::stepBegin("AnimateVisitor");
    AnimateVisitor act(params, dt);
    node->execute ( act );
    sofa::helper::AdvancedTimer::stepEnd("AnimateVisitor");


    sofa::helper::AdvancedTimer::stepBegin("UpdateSimulationContextVisitor");
    node->setTime ( startTime + dt );
    node->execute< UpdateSimulationContextVisitor >(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateSimulationContextVisitor");

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor propagateEventVisitor ( params, &ev );
        node->execute ( propagateEventVisitor );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    node->execute< UpdateMappingVisitor >(params);
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor propagateEventVisitor ( params , &ev );
        node->execute ( propagateEventVisitor );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

    if (d_computeBoundingBox.getValue())
    {
        sofa::helper::ScopedAdvancedTimer timer("UpdateBBox");
        node->execute< UpdateBoundingBoxVisitor >(params);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif


}


} // namespace simulation

} // namespace sofa
