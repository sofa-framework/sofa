#include "AnimationLoopTasks.h"

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
//#include <sofa/helper/system/atomic.h>

#include <thread>

namespace sofa
{
    
    namespace simulation
    {
        
        
        
        StepTask::StepTask(core::behavior::BaseAnimationLoop* aloop, const double t, CpuTask::Status* status)
        : CpuTask(status)
        , animationloop(aloop)
        , dt(t)
        {
        }
        
        StepTask::~StepTask()
        {
        }
        
        
        Task::MemoryAlloc StepTask::run()
        {
            animationloop->step( core::ExecParams::defaultInstance(), dt);
            return Task::MemoryAlloc::Dynamic;
        }        
        
    } // namespace simulation
    
} // namespace sofa




