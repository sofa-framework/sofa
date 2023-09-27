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
#include <sofa/component/animationloop/MultiTagAnimationLoop.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/helper/ScopedAdvancedTimer.h>

#include <sofa/simulation/mechanicalvisitor/MechanicalResetConstraintVisitor.h>
using sofa::simulation::mechanicalvisitor::MechanicalResetConstraintVisitor;

using namespace sofa::simulation;

namespace sofa::component::animationloop
{

int MultiTagAnimationLoopClass = core::RegisterObject("Simple animation loop that given a list of tags, animate the graph one tag after another.")
        .add< MultiTagAnimationLoop >()
        .addAlias("MultiTagMasterSolver")
        ;

MultiTagAnimationLoop::MultiTagAnimationLoop()
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
    auto node = dynamic_cast<sofa::simulation::Node*>(this->l_node.get());

    SCOPED_TIMER_VARNAME(animationStepTimer, "AnimationStep");

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printNode("Step");
#endif

    {
        AnimateBeginEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    SReal startTime = node->getTime();

    BehaviorUpdatePositionVisitor beh(params , dt);
    node->execute ( beh );

    UpdateInternalDataVisitor uid(params);
    node->execute ( uid );

    sofa::core::objectmodel::TagSet::iterator it;

    sofa::core::ConstraintParams cparams(*params);

    for (it = tagList.begin(); it != tagList.end(); ++it)
    {
        this->addTag (*it);

        dmsg_info() << "begin constraints reset" ;
        MechanicalResetConstraintVisitor(&cparams).execute(node);
        dmsg_info() << "end constraints reset" ;

        dmsg_info() << "begin collision for tag: "<< *it ;
        computeCollision(params);
        dmsg_info() << "step, end collision" ;
        dmsg_info() << "step, begin integration  for tag: "<< *it ;
        integrate(params, dt);
        dmsg_info() << "end integration" << msgendl;

        this->removeTag (*it);
    }

    node->setTime ( startTime + dt );
    node->execute<UpdateSimulationContextVisitor>(params);  // propagate time

    {
        AnimateEndEvent ev ( dt );
        PropagateEventVisitor act ( params, &ev );
        node->execute ( act );
    }

    //Visual Information update: Ray Pick add a MechanicalMapping used as VisualMapping
    {
        SCOPED_TIMER_VARNAME(updateMappingTimer, "UpdateMapping");
        node->execute<UpdateMappingVisitor>(params);
    }
    {
        UpdateMappingEndEvent ev ( dt );
        PropagateEventVisitor act ( params , &ev );
        node->execute ( act );
    }

    if (d_computeBoundingBox.getValue())
    {
        SCOPED_TIMER("UpdateBBox");
        node->execute<UpdateBoundingBoxVisitor>(params);
    }

#ifdef SOFA_DUMP_VISITOR_INFO
    simulation::Visitor::printCloseNode("Step");
#endif
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

} // namespace sofa::component::animationloop
