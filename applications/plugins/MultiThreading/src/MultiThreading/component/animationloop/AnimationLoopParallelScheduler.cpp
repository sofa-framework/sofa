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
#include <MultiThreading/AnimationLoopTasks.h>
#include <sofa/simulation/InitTasks.h>
#include <MultiThreading/DataExchange.h>

#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ConstraintSolver.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/core/CollisionModel.h>
#include <sofa/simulation/PrintVisitor.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/ExportGnuplotVisitor.h>
#include <sofa/simulation/InitVisitor.h>
#include <sofa/simulation/AnimateVisitor.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/CollisionBeginEvent.h>
#include <sofa/simulation/CollisionEndEvent.h>
#include <sofa/simulation/UpdateContextVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
#include <sofa/simulation/ResetVisitor.h>
#include <sofa/simulation/VisualVisitor.h>
#include <sofa/simulation/ExportVisualModelOBJVisitor.h>
#include <sofa/simulation/WriteStateVisitor.h>
#include <sofa/simulation/XMLPrintVisitor.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/UpdateInternalDataVisitor.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/UpdateMappingEndEvent.h>
#include <sofa/simulation/CleanupVisitor.h>
#include <sofa/simulation/DeleteVisitor.h>
#include <sofa/simulation/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/common/xml/NodeElement.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PipeProcess.h>
#include <sofa/helper/AdvancedTimer.h>

#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/MainTaskSchedulerFactory.h>

#include <cstdlib>
#include <cmath>
#include <algorithm>

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

    for (const auto& it : gnode->child)
    {
        if ( sofa::core::behavior::BaseAnimationLoop* aloop = it->getAnimationLoop() )
        {
            m_taskScheduler->addTask(new StepTask(aloop, dt, &status));
        }
    }

    m_taskScheduler->workUntilDone(&status);

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
