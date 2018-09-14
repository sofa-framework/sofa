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
        
        
        
        StepTask::StepTask(core::behavior::BaseAnimationLoop* aloop, const double t, Task::Status* pStatus)
        : Task(pStatus)
        , animationloop(aloop)
        , dt(t)
        {
        }
        
        StepTask::~StepTask()
        {
        }
        
        
        bool StepTask::run(WorkerThread* )
        {
            animationloop->step( core::ExecParams::defaultInstance(), dt);
            return true;
        }        
        
    } // namespace simulation
    
} // namespace sofa




