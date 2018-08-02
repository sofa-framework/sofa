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

        static HDC glGlobalDevice = nullptr;
        static thread_local HGLRC glGlobalContext = nullptr;


        InitPerThreadDataTask::InitPerThreadDataTask(std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus)
            : Task(pStatus), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
        {}

        InitPerThreadDataTask::~InitPerThreadDataTask()
        {
        }

        bool InitPerThreadDataTask::run(WorkerThread*)
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



        InitOGLcontextTask::InitOGLcontextTask(HDC& glDevice, HGLRC& workerThreadContext, std::atomic<int>* atomicCounter, std::mutex* mutex, Task::Status* pStatus)
            : Task(pStatus), _glDevice(glDevice), _workerThreadContext(workerThreadContext), IdFactorygetIDMutex(mutex), _atomicCounter(atomicCounter)
        {}

        InitOGLcontextTask::~InitOGLcontextTask()
        {
        }

        bool InitOGLcontextTask::run(sofa::simulation::WorkerThread*)
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

        bool DeleteOGLcontextTask::run(sofa::simulation::WorkerThread*)
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

        // temp remove this function to use the global one
        void initThreadLocalData()
        {
            std::atomic<int> atomicCounter;
            atomicCounter = TaskScheduler::getInstance().size();

            std::mutex  InitThreadSpecificMutex;

            Task::Status status;

            const int nbThread = TaskScheduler::getInstance().size();
            WorkerThread* thread = WorkerThread::getCurrent();

            for (int i = 0; i<nbThread; ++i)
            {
                thread->addTask(new InitPerThreadDataTask(&atomicCounter, &InitThreadSpecificMutex, &status));
            }

            thread->workUntilDone(&status);

            return;
        }

        // temp remove this function to use the global one
        void initOGLcontext()
        {
            HWND hwin = FindWindow(0, L"GLEWTest");
            glGlobalDevice = wglGetCurrentDC();// GetDC(hwin);
                                               //main thread context
                                               // share this context with worker thread contexts
            glGlobalContext = wglGetCurrentContext();// wglCreateContext(glGlobalDevice);
            HGLRC mainContext = glGlobalContext;

            // drop the main context before sharing the context
            // http://hacksoflife.blogspot.com/2008/02/creating-opengl-objects-in-second.html
            wglMakeCurrent(NULL, NULL);

            std::atomic<int> atomicCounter;
            atomicCounter = sofa::simulation::TaskScheduler::getInstance().size() - 1;

            std::mutex  InitThreadSpecificMutex;

            sofa::simulation::Task::Status status;

            const int nbWorkerThread = sofa::simulation::TaskScheduler::getInstance().size() - 1;
            sofa::simulation::WorkerThread* thread = sofa::simulation::WorkerThread::getCurrent();

            for (int i = 0; i<nbWorkerThread; ++i)
            {
                // creqte a new context and share it with the main thread context
                HGLRC workerThreadContexts = wglCreateContext(glGlobalDevice);
                wglShareLists(mainContext, workerThreadContexts);

                thread->addTask(new InitOGLcontextTask(glGlobalDevice, workerThreadContexts, &atomicCounter, &InitThreadSpecificMutex, &status));
            }

            // wait for worker thread to complete the per-thread task
            while (status.isBusy())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            // set the main context back after sharing the context
            // http://hacksoflife.blogspot.com/2008/02/creating-opengl-objects-in-second.html
            wglMakeCurrent(glGlobalDevice, glGlobalContext);


            return;
        }

        // temp remove this function to use the global one
        void deleteOGLcontext()
        {
            std::atomic<int> atomicCounter;
            atomicCounter = sofa::simulation::TaskScheduler::getInstance().size() - 1;

            std::mutex  InitThreadSpecificMutex;

            sofa::simulation::Task::Status status;

            const int nbWorkerThread = sofa::simulation::TaskScheduler::getInstance().size() - 1;
            sofa::simulation::WorkerThread* thread = sofa::simulation::WorkerThread::getCurrent();

            for (int i = 0; i<nbWorkerThread; ++i)
            {
                thread->addTask(new DeleteOGLcontextTask(&atomicCounter, &InitThreadSpecificMutex, &status));
            }

            // wait for worker thread to complete the per-thread task
            while (status.isBusy())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            return;
        }

    } // namespace simulation

} // namespace sofa