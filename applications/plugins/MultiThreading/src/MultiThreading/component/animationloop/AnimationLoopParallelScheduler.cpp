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
#include <MultiThreading/component/animationloop/AnimationLoopParallelScheduler.h>

#include <sofa/simulation/TaskScheduler.h>
#include <sofa/simulation/InitTasks.h>
#include <MultiThreading/DataExchange.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/PrintVisitor.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/common/xml/NodeElement.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>
#include <sofa/simulation/ParallelForEach.h>

namespace multithreading::component::animationloop
{

int AnimationLoopParallelSchedulerClass = sofa::core::RegisterObject("parallel animation loop, using intel tbb library")
        .add< AnimationLoopParallelScheduler >()
        ;

AnimationLoopParallelScheduler::AnimationLoopParallelScheduler(sofa::simulation::Node* _gnode)
    : Inherit()
    , gnode(_gnode)
{}

AnimationLoopParallelScheduler::~AnimationLoopParallelScheduler() = default;

void AnimationLoopParallelScheduler::init()
{
    if (!gnode)
        gnode = dynamic_cast<sofa::simulation::Node*>(this->getContext());

    initTaskScheduler();
}

void AnimationLoopParallelScheduler::bwdInit()
{
    sofa::simulation::initThreadLocalData();
}

void AnimationLoopParallelScheduler::reinit()
{
    this->reinitTaskScheduler();
}

void AnimationLoopParallelScheduler::cleanup()
{
    this->stopTaskSchduler();
}

void AnimationLoopParallelScheduler::step(const sofa::core::ExecParams* params, SReal dt)
{
    if (dt == 0)
        dt = this->gnode->getDt();

    sofa::simulation::CpuTask::Status status;

    sofa::simulation::parallelForEach(*m_taskScheduler,
        gnode->child.begin(), gnode->child.end(),
        [dt](const auto& node)
        {
            if ( sofa::core::behavior::BaseAnimationLoop* aloop = node->getAnimationLoop() )
            {
                aloop->step(sofa::core::ExecParams::defaultInstance(), dt);
            }
        });

    double startTime = gnode->getTime();
    gnode->setTime ( startTime + dt );

    // exchange data event
    sofa::core::DataExchangeEvent ev ( dt );
    sofa::simulation::PropagateEventVisitor act ( params, &ev );
    gnode->execute ( act );


    // it doesn't call the destructor
    //task_pool.purge_memory();
}

} // namespace multithreading::component::animationloop
