/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef MultiThreadingTask_h__
#define MultiThreadingTask_h__

#include <sofa/config.h>

#include <atomic>
#include <mutex>


namespace sofa
{
	namespace simulation
    {


        /** Task class interface    */
        class SOFA_SIMULATION_CORE_API Task
        {
        public:
            
            // Task Status class interface used to synchronize tasks
            class Status
            {
            public:
                virtual ~Status() {}
                virtual bool isBusy() const = 0;
                virtual int setBusy(bool busy) const = 0;
            };
            
            // Task Allocator class interface used to allocate tasks
            class Allocator
            {
            public:
                virtual void* allocate(std::size_t sz) = 0;
                
                virtual void free(void* ptr, std::size_t sz) = 0;
            };
            
            
            
            Task(const Task::Status* status, int scheduledThread);
            
            virtual ~Task();
            
            
            enum MemoryAlloc
            {
                Stack     = 1 << 0,
                Dynamic   = 1 << 1,
                Static    = 1 << 2,
            };
            
            
            // Task interface: override these two functions
            virtual MemoryAlloc run() = 0;
            
            
            static void* operator new (std::size_t sz)
            {
                return _allocator->allocate(sz);
            }
            
            // when c++14 is available delete the void  operator delete  (void* ptr)
            // and define the void operator delete  (void* ptr, std::size_t sz)
            static void  operator delete  (void* ptr)
            {
                _allocator->free(ptr, 0);
            }
            
            // only available in c++14.
            static void operator delete  (void* ptr, std::size_t sz)
            {
                _allocator->free(ptr, sz);
            }
            
            // no array new and delete operators
            static void* operator new[](std::size_t sz) = delete;
            
            // visual studio 2015 complains about the = delete but it doens't explain where this operator is call
            // no problem with other sompilers included visual studio 2017
            //static void operator delete[](void* ptr) = delete;
            
            virtual const Task::Status* getStatus(void) const = 0;
            
            int getScheduledThread() const { return m_scheduledThread; }
            
            static Task::Allocator* getAllocator() { return _allocator; }
            
            static void setAllocator(Task::Allocator* allocator) { _allocator = allocator; }
            
        protected:
            
            const Task::Status*    m_status;
            
            int m_scheduledThread;
            
        public:
            int m_id;
            
        private:
            
            static Task::Allocator * _allocator;
        };
        
        

        /**  Base class to implement a CPU task
         *   all the tasks running on the CPU should inherits from this class
         */
        class SOFA_SIMULATION_CORE_API CpuTask : public Task
        {
        public:

            /** CPU Task Status class definition:
             *  used to synchronize CPU tasks  */
            class Status : public Task::Status
            {
            public:
                Status() : m_busy(0) {}

                virtual bool isBusy() const override final
                {
                    return (m_busy.load(std::memory_order_relaxed) > 0);
                }

                virtual int setBusy(bool busy) const override final
                {
                    if (busy)
                    {
                        return m_busy.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        return m_busy.fetch_sub(1, std::memory_order_relaxed);
                    }
                }

            private:
                mutable std::atomic<int> m_busy;
            };


            virtual const CpuTask::Status* getStatus(void) const override final 
                { return dynamic_cast<const CpuTask::Status*>(m_status); }


        public:
            
            CpuTask(const CpuTask::Status* status, int scheduledThread = -1);

            virtual ~CpuTask();

        };




        // This task is called once by each thread used by the TasScheduler
        // this is useful to initialize the thread specific variables
        class SOFA_SIMULATION_CORE_API ThreadSpecificTask : public CpuTask
        {

        public:

            ThreadSpecificTask(std::atomic<int>* atomicCounter, std::mutex* mutex, const CpuTask::Status* status);

            ~ThreadSpecificTask() override;

            MemoryAlloc run() final;


        private:

            virtual bool runThreadSpecific() { return true; }

            virtual bool runCriticalThreadSpecific() { return true; }


            std::atomic<int>* m_atomicCounter;
            std::mutex*     m_threadSpecificMutex;
        };


	} // namespace simulation

} // namespace sofa



#endif // MultiThreadingTask_h__
