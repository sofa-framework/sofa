#include <sofa/simulation/ParallelFor.h>

namespace sofa
{
	namespace simulation
	{
       
        ForTask::Range ForTask::Range::split(Range& r)
        {
            assert(r.is_divisible()); // for range is not divisible
            Range second_half(r._first + (r._last - r._first) / 2u, r._last, r._grainsize);
            r._last = second_half._first;
            return second_half;
        }



        class InternalForTask : public simulation::Task
        {
        public:

            InternalForTask(const ForTask* forTask, const ForTask::Range& range, const Task::Status* status)
                : Task(status)
                , _range(range)
                , _forTask(forTask)
            {}


            virtual ~InternalForTask() {}

            virtual bool run() final
            {
                if (_range.is_divisible())
                {
                    InternalForTask* newInternalTask = new InternalForTask(_forTask, ForTask::Range::split(_range), _status);
                    simulation::TaskScheduler::getInstance()->addTask(newInternalTask);
                    // keep running
                    this->run();
                }
                else
                {
                    _forTask->operator()(_range);
                }

                return true;
            }

        public:
            const ForTask* _forTask;
            ForTask::Range _range;
        };

        void ParallelFor(ForTask& task, const ForTask::Range& range)
        {
            simulation::Task::Status status;
            InternalForTask* internalTask = new InternalForTask(&task, range, &status);
            simulation::TaskScheduler* currentScheduler = TaskScheduler::getInstance();
            currentScheduler->addTask(internalTask);
            currentScheduler->workUntilDone(&status);
        }


	} // namespace simulation

} // namespace sofa
