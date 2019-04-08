#include <sofa/simulation/InitTasks.h>

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>


namespace sofa
{

    namespace simulation
    {
        
        InitPerThreadDataTask::InitPerThreadDataTask(std::atomic<int>* atomicCounter, std::mutex* mutex, CpuTask::Status* status)
        : CpuTask(status), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
        {}
        
        InitPerThreadDataTask::~InitPerThreadDataTask()
        {
        }
        
        Task::MemoryAlloc InitPerThreadDataTask::run()
        {
            
            core::ExecParams::defaultInstance();
            
            core::ConstraintParams::defaultInstance();
            
            core::MechanicalParams::defaultInstance();
            
            core::visual::VisualParams::defaultInstance();
            
            {
                // to solve IdFactory<Base>::getID() problem in AdvancedTimer functions
                std::lock_guard<std::mutex> lock(*IdFactorygetIDMutex);
                
                //spinMutexLock lock( IdFactorygetIDMutex );
                
                helper::AdvancedTimer::begin("Animate");
                helper::AdvancedTimer::end("Animate");
            }
            
            _atomicCounter->fetch_sub(1, std::memory_order_acq_rel);
            
            while (_atomicCounter->load(std::memory_order_relaxed) > 0)
            {
                // yield while waiting  
                std::this_thread::yield();
            }
            return Task::MemoryAlloc::Dynamic;
        }
        
        
        // temp remove this function to use the global one
        void initThreadLocalData()
        {
            std::atomic<int> atomicCounter;
            TaskScheduler* scheduler = TaskScheduler::getInstance();
            atomicCounter = scheduler->getThreadCount();
            
            std::mutex  InitThreadSpecificMutex;
            
            CpuTask::Status status;
            const int nbThread = scheduler->getThreadCount();
            
            for (int i = 0; i<nbThread; ++i)
            {
                scheduler->addTask(new InitPerThreadDataTask(&atomicCounter, &InitThreadSpecificMutex, &status));
            }
            
            scheduler->workUntilDone(&status);
            
            return;
        }
        
        
    } // namespace simulation

} // namespace sofa
