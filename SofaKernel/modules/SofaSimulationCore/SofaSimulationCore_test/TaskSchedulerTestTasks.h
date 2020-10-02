#include <sofa/simulation/Task.h>


namespace sofa
{

    // compute recursively the Fibonacci number for input N  O(~1.6 exp(N)) 
    // this is implemented to test the task scheduler generating super lightweight tasks and not for performance
    class FibonacciTask : public simulation::CpuTask
    {
    public:
        FibonacciTask(const int64_t N, int64_t* const sum, simulation::CpuTask::Status* status)
        : CpuTask(status)
        , _N(N)
        , _sum(sum)
        {}
        
        ~FibonacciTask() override { }
        
        MemoryAlloc run() final;
        
    private:
        
        const int64_t _N;
        int64_t* const _sum;
    };
    
    
    // compute recursively the sum of integers from first to last
    // this is implemented to test the task scheduler generating super lightweight tasks and not for performance
    class IntSumTask : public simulation::CpuTask
    {
    public:
        IntSumTask(const int64_t first, const int64_t last, int64_t* const sum, simulation::CpuTask::Status* status)
        : CpuTask(status) 
        , _first(first)
        , _last(last)
        , _sum(sum)
        {}
        
        ~IntSumTask() override {}
        
        MemoryAlloc run() final;
        
        
    private:
        
        const int64_t _first;
        const int64_t _last;
        int64_t* const _sum;
        
    };
} // namespace sofa
