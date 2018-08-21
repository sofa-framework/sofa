#include "InitTasks.h"

#include <sofa/core/behavior/BaseAnimationLoop.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/ConstraintParams.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/helper/AdvancedTimer.h>
//#include <sofa/helper/system/atomic.h>


namespace sofa
{

    namespace simulation
    {

        InitPerThreadDataTask::InitPerThreadDataTask(std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus)
            : Task(pStatus), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
        {}

        InitPerThreadDataTask::~InitPerThreadDataTask()
        {
        }

        bool InitPerThreadDataTask::run()
        {

            core::ExecParams::defaultInstance();

            core::ConstraintParams::defaultInstance();

            core::MechanicalParams::defaultInstance();

            core::visual::VisualParams::defaultInstance();

            // init curTimerThread
            //helper::getCurTimer();
            //std::stack<AdvancedTimer::IdTimer>& getCurTimer();
            {
                // to solve IdFactory<Base>::getID() problem in AdvancedTimer functions
                std::lock_guard<std::mutex> lock(*IdFactorygetIDMutex);

                //spinMutexLock lock( IdFactorygetIDMutex );

                helper::AdvancedTimer::begin("Animate");
                helper::AdvancedTimer::end("Animate");
                //helper::AdvancedTimer::stepBegin("AnimationStep");
                //helper::AdvancedTimer::stepEnd("AnimationStep");
            }


            //BOOST_INTERLOCKED_DECREMENT( mAtomicCounter );
            //BOOST_COMPILER_FENCE;

            _atomicCounter->fetch_sub(1, std::memory_order_acq_rel);


            while (_atomicCounter->load(std::memory_order_relaxed) > 0)
            {
                // yield while waiting  
                std::this_thread::yield();
            }
            return true;
        }

        
        // temp remove this function to use the global one
        void initThreadLocalData()
        {
            std::atomic<int> atomicCounter;
            atomicCounter = TaskScheduler::getInstance()->getThreadCount();

            std::mutex  InitThreadSpecificMutex;

            Task::Status status;

            TaskScheduler* scheduler = TaskScheduler::getInstance();
            const int nbThread = scheduler->getThreadCount();

            for (int i = 0; i<nbThread; ++i)
            {
                scheduler->addTask(new InitPerThreadDataTask(&atomicCounter, &InitThreadSpecificMutex, &status));
            }

            scheduler->workUntilDone(&status);

            return;
        }
        

#ifdef _WIN32
        

        static HDC glGlobalDevice = nullptr;
        static thread_local HGLRC glGlobalContext = nullptr;
        
        
        InitOGLcontextTask::InitOGLcontextTask(HDC& glDevice, HGLRC& workerThreadContext, std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus)
            : Task(pStatus), _glDevice(glDevice), _workerThreadContext(workerThreadContext), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
        {}

        InitOGLcontextTask::~InitOGLcontextTask()
        {
        }

        bool InitOGLcontextTask::run()
        {
            //glGlobalContext = wglCreateContext(_hdc);
            //wglShareLists(_mainContext, glGlobalContext);
            glGlobalContext = _workerThreadContext;
            wglMakeCurrent(_glDevice, _workerThreadContext);

            _atomicCounter->fetch_sub(1, std::memory_order_acq_rel);
            while (_atomicCounter->load(std::memory_order_relaxed) > 0)
            {
                // yield while waiting  
                std::this_thread::yield();
            }
            return true;
        }


        DeleteOGLcontextTask::DeleteOGLcontextTask(std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus)
            : Task(pStatus), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
        {}

        DeleteOGLcontextTask::~DeleteOGLcontextTask()
        {
        }

        bool DeleteOGLcontextTask::run()
        {
            wglDeleteContext(glGlobalContext);

            _atomicCounter->fetch_sub(1, std::memory_order_acq_rel);
            while (_atomicCounter->load(std::memory_order_relaxed) > 0)
            {
                // yield while waiting  
                std::this_thread::yield();
            }
            return true;
        }

#endif // _WIN32
        
        
    } // namespace simulation

} // namespace sofa
