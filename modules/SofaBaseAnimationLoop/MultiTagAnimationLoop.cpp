/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaBaseAnimationLoop/MultiTagAnimationLoop.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
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

        if (this->f_printLog.getValue()) sout << "MultiTagAnimationLoop::step, begin constraints reset" << sendl;
        sofa::simulation::MechanicalResetConstraintVisitor(params).execute(this->getContext());
        if (this->f_printLog.getValue()) sout << "MultiTagAnimationLoop::step, end constraints reset" << sendl;
        if (this->f_printLog.getValue()) sout << "MultiTagAnimationLoop::step, begin collision for tag: "<< *it << sendl;
        computeCollision(params);
        if (this->f_printLog.getValue()) sout << "MultiTagAnimationLoop::step, end collision" << sendl;
        if (this->f_printLog.getValue()) sout << "MultiTagAnimationLoop::step, begin integration  for tag: "<< *it << sendl;
        integrate(params, dt);
        if (this->f_printLog.getValue()) sout << "MultiTagAnimationLoop::step, end integration" << sendl;

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

