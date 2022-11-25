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
#include <MultiThreading/AnimationLoopParallelScheduler.h>

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

namespace sofa::simulation
{

int AnimationLoopParallelSchedulerClass = core::RegisterObject("parallel animation loop, using intel tbb library")
        .add< AnimationLoopParallelScheduler >()
        ;

AnimationLoopParallelScheduler::AnimationLoopParallelScheduler(simulation::Node* _gnode)
    : Inherit()
    , schedulerName(initData(&schedulerName, "scheduler", "name of the scheduler to use"))
    , threadNumber(initData(&threadNumber, (unsigned int)0, "threadNumber", "number of thread") )
    , mNbThread(0)
    , gnode(_gnode)
    , _taskScheduler(nullptr)
{}

AnimationLoopParallelScheduler::~AnimationLoopParallelScheduler() = default;

void AnimationLoopParallelScheduler::init()
{
    if (!gnode)
        gnode = dynamic_cast<simulation::Node*>(this->getContext());

    if ( threadNumber.getValue() )
    {
        mNbThread = threadNumber.getValue();
    }

    if (schedulerName.isSet())
    {
        _taskScheduler = MainTaskSchedulerFactory::createInRegistry(schedulerName.getValue() );
        if (!_taskScheduler)
        {
            msg_error() << "'" << schedulerName.getValue()
                << "' is not a valid name for a task scheduler. Falling back to the default "
                "task scheduler. The list of available schedulers is: ["
                << sofa::helper::join(MainTaskSchedulerFactory::getAvailableSchedulers(), ',')
                << "]";
        }
    }

    if (!_taskScheduler)
    {
        _taskScheduler = MainTaskSchedulerFactory::createInRegistry();
    }

    if (_taskScheduler)
    {
        _taskScheduler->init( mNbThread );
    }
    else
    {
        this->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Invalid);
    }
}

void AnimationLoopParallelScheduler::bwdInit()
{
    initThreadLocalData();
}

void AnimationLoopParallelScheduler::reinit()
{
    if ( threadNumber.getValue() != _taskScheduler->getThreadCount() )
    {
        mNbThread = threadNumber.getValue();
        _taskScheduler->init(mNbThread);
        initThreadLocalData();
    }
}

void AnimationLoopParallelScheduler::cleanup()
{
    _taskScheduler->stop();
}

void AnimationLoopParallelScheduler::step(const core::ExecParams* params, SReal dt)
{
    if (dt == 0)
        dt = this->gnode->getDt();

    simulation::CpuTask::Status status;

    for (const auto& it : gnode->child)
    {
        if ( core::behavior::BaseAnimationLoop* aloop = it->getAnimationLoop() )
        {
            _taskScheduler->addTask(new StepTask(aloop, dt, &status));
        }
    }

    _taskScheduler->workUntilDone(&status);

    double startTime = gnode->getTime();
    gnode->setTime ( startTime + dt );

    // exchange data event
    core::DataExchangeEvent ev ( dt );
    PropagateEventVisitor act ( params, &ev );
    gnode->execute ( act );


    // it doesn't call the destructor
    //task_pool.purge_memory();
}

} // namespace sofa::simulation
