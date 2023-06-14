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


namespace sofa::simulation
{

int DefaultAnimationLoopClass = core::RegisterObject("Simulation loop to use in scene without constraints nor contact.")
                                .add<DefaultAnimationLoop>()
                                .addDescription(R"(
This loop does the following steps:
- build and solve all linear systems in the scene : collision and time integration to compute the new values of the dofs
- update the context (dt++)
- update the mappings
- update the bounding box (volume covering all objects of the scene))");

DefaultAnimationLoop::DefaultAnimationLoop(simulation::Node* _gnode)
    : Inherit()
    , gnode(_gnode)
{
    //assert(gnode);
}

DefaultAnimationLoop::~DefaultAnimationLoop() = default;

void DefaultAnimationLoop::init()
{
    if (!gnode)
    {
        gnode = dynamic_cast<simulation::Node*>(this->getContext());
    }
}

void DefaultAnimationLoop::setNode(simulation::Node* n)
{
    gnode = n;
}

void DefaultAnimationLoop::behaviorUpdatePosition(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("BehaviorUpdatePositionVisitor");
    BehaviorUpdatePositionVisitor beh(params, dt);
    gnode->execute(beh);
}

void DefaultAnimationLoop::updateInternalData(const core::ExecParams* params) const
{
    sofa::helper::ScopedAdvancedTimer timer("UpdateInternalDataVisitor");
    gnode->execute<UpdateInternalDataVisitor>(params);
}

void DefaultAnimationLoop::animate(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("AnimateVisitor");
    AnimateVisitor act(params, dt);
    gnode->execute(act);
}

void DefaultAnimationLoop::updateSimulationContext(const core::ExecParams* params, const SReal dt, const SReal startTime) const
{
    sofa::helper::ScopedAdvancedTimer timer("UpdateSimulationContextVisitor");
    gnode->setTime(startTime + dt);
    gnode->execute<UpdateSimulationContextVisitor>(params);
}

void DefaultAnimationLoop::animateEndEvent(const core::ExecParams* params, const SReal dt) const
{
    AnimateEndEvent ev(dt);
    PropagateEventVisitor propagateEventVisitor(params, &ev);
    gnode->execute(propagateEventVisitor);
}

void DefaultAnimationLoop::updateMapping(const core::ExecParams* params, const SReal dt) const
{
    sofa::helper::ScopedAdvancedTimer timer("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    gnode->execute<UpdateMappingVisitor>(params);
    {
        UpdateMappingEndEvent ev(dt);
        PropagateEventVisitor propagateEventVisitor(params, &ev);
        gnode->execute(propagateEventVisitor);
    }
}

void DefaultAnimationLoop::computeBoundingBox(const core::ExecParams* params) const
{
    if (d_computeBoundingBox.getValue())
    {
        sofa::helper::ScopedAdvancedTimer timer("UpdateBBox");
        gnode->execute<UpdateBoundingBoxVisitor>(params);
    }
}

void DefaultAnimationLoop::animateBeginEvent(const core::ExecParams* params, const SReal dt) const
{
    AnimateBeginEvent ev(dt);
    PropagateEventVisitor act(params, &ev);
    gnode->execute(act);
}

void DefaultAnimationLoop::step(const core::ExecParams* params, SReal dt)
{
    if (dt == 0)
    {
        dt = this->gnode->getDt();
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    animateBeginEvent(params, dt);

    const SReal startTime = gnode->getTime();

    behaviorUpdatePosition(params, dt);
    updateInternalData(params);
    animate(params, dt);
    updateSimulationContext(params, dt, startTime);
    animateEndEvent(params, dt);
    updateMapping(params, dt);
    computeBoundingBox(params);

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
}


} // namespace sofa
