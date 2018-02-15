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
#include <SofaGeneralAnimationLoop/MultiTagAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <math.h>
#include <iostream>


using namespace sofa::simulation;

namespace sofa
{

namespace component
{

namespace animationloop
{

int MultiTagAnimationLoopClass = core::RegisterObject("Simple animation loop that given a list of tags, animate the graph one tag after another.")
        .add< MultiTagAnimationLoop >()
        .addAlias("MultiTagMasterSolver")
        ;

SOFA_DECL_CLASS(MultiTagAnimationLoop);

MultiTagAnimationLoop::MultiTagAnimationLoop(simulation::Node* gnode)
    : Inherit(gnode)
{
}

MultiTagAnimationLoop::~MultiTagAnimationLoop()
{
}

void MultiTagAnimationLoop::init()
{
    tagList = this->getTags();
    sofa::core::objectmodel::TagSet::iterator it;

    for (it = tagList.begin(); it != tagList.end(); ++it)
        this->removeTag (*it);
}



void MultiTagAnimationLoop::step(const sofa::core::ExecParams* params, SReal dt)
{
    sofa::helper::AdvancedTimer::stepBegin("AnimationStep");
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    SReal startTime = this->gnode->getTime();

    BehaviorUpdatePositionVisitor beh(params , dt);
    this->gnode->execute ( beh );

    sofa::core::objectmodel::TagSet::iterator it;

    for (it = tagList.begin(); it != tagList.end(); ++it)
    {
        this->addTag (*it);

        dmsg_info() << "begin constraints reset" ;
        sofa::simulation::MechanicalResetConstraintVisitor(params).execute(this->getContext());
        dmsg_info() << "end constraints reset" ;

        dmsg_info() << "begin collision for tag: "<< *it ;
        computeCollision(params);
        dmsg_info() << "step, end collision" ;
        dmsg_info() << "step, begin integration  for tag: "<< *it ;
        integrate(params, dt);
        dmsg_info() << "end integration" << sendl;

        this->removeTag (*it);
    }

    this->gnode->setTime ( startTime + dt );
    this->gnode->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        this->gnode->execute ( act );
    }

    sofa::helper::AdvancedTimer::stepBegin("UpdateMapping");
    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    this->gnode->execute<UpdateMappingVisitor>(params);
    sofa::helper::AdvancedTimer::step("UpdateMappingEndEvent");
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        this->gnode->execute ( act );
    }
    sofa::helper::AdvancedTimer::stepEnd("UpdateMapping");

#ifndef SOFA_NO_UPDATE_BBOX
    sofa::helper::AdvancedTimer::stepBegin("UpdateBBox");
    this->gnode->execute<UpdateBoundingBoxVisitor>(params);
    sofa::helper::AdvancedTimer::stepEnd("UpdateBBox");
#endif
#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif

    sofa::helper::AdvancedTimer::stepEnd("AnimationStep");
}

void MultiTagAnimationLoop::clear()
{
    if (!tagList.empty())
    {
        sofa::core::objectmodel::TagSet::iterator it;
        for (it = tagList.begin(); it != tagList.end(); ++it)
            this->addTag (*it);

        tagList.clear();
    }
}



} // namespace animationloop

} // namespace component

} // namespace sofa

