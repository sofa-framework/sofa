#include <sofa/simulation/Task.h>

#include <cassert>
#include <thread>


namespace sofa
{
	namespace simulation
	{
        

        Task::Allocator* Task::_allocator = nullptr;


		Task::Task(const Task::Status* status, int scheduledThread)
			: m_scheduledThread(scheduledThread)
            , m_status(status)
            , m_id(0)
		{            
		}

		Task::~Task()
		{
		}
        
        
        CpuTask::CpuTask(const CpuTask::Status* status, int scheduledThread)
        : Task(status, scheduledThread)
        {
        }
        
        CpuTask::~CpuTask()
        {
        }
        
        
        
        
        ThreadSpecificTask::ThreadSpecificTask(std::atomic<int>* atomicCounter, std::mutex* mutex, const CpuTask::Status* status )
            : CpuTask(status)
            , m_atomicCounter(atomicCounter)
            , m_threadSpecificMutex(mutex)
        {}

        ThreadSpecificTask::~ThreadSpecificTask()
        {
        }

        Task::MemoryAlloc ThreadSpecificTask::run()
        {

            runThreadSpecific();

            {
                std::lock_guard<std::mutex> lock(*m_threadSpecificMutex);
                runCriticalThreadSpecific();
            }

            m_atomicCounter->fetch_sub(1, std::memory_order_acq_rel);

            while(m_atomicCounter->load(std::memory_order_relaxed) > 0)
            {
                // yield while waiting
                std::this_thread::yield();
            }
            return Task::MemoryAlloc::Stack;
        }

	
	} // namespace simulation

} // namespace sofa
