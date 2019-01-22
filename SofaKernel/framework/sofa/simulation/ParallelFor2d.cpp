#include <sofa/simulation/ParallelFor2d.h>

namespace sofa
{
	namespace simulation
	{


        class InternalForTask2d : public simulation::Task
        {
        public:

            InternalForTask2d(const ForTask2d* forTask, const ForTask2d::Range& range, const Task::Status* status)
                : Task(status)
                , _range(range)
                , _forTask(forTask)
            {}


            virtual ~InternalForTask2d() {}

            virtual bool run() final
            {
                if (_partition & ForTask2d::Partition::simple)
                {
                    simplePartition();
                }
                else if (_partition & ForTask2d::Partition::avoid_shared_data)
                {
                    preventSharedDataPartition();
                }

                return true;
            }

        private:

            void simplePartition()
            {
                if (_range.is_divisible())
                {
                    Task::Status synchStatus;
                    InternalForTask2d* newInternalTask = new InternalForTask2d(_forTask, ForTask2d::Range::split(_range), &synchStatus);
                    simulation::TaskScheduler::getInstance()->addTask(newInternalTask);
                    // keep running
                    this->run();
                    simulation::TaskScheduler::getInstance()->workUntilDone(&synchStatus);
                }
                else
                {
                    _forTask->operator()(_range);
                }
            }
            
            void preventSharedDataPartition()
            {
                ForTask2d::Range ranges[3];
                ranges[2] = _range;
                int newRangeCounter = 0;

                if (_range.rows().is_divisible())
                {
                    ranges[newRangeCounter] = ForTask2d::Range(ForTask::Range::split(_range._rows), ranges[2].cols());
                    ++newRangeCounter;
                }
                if (_range.cols().is_divisible())
                {
                    ranges[newRangeCounter] = ForTask2d::Range(ranges[2].rows(), ForTask::Range::split(_range._cols));
                    ++newRangeCounter;
                }

                if (newRangeCounter == 2)
                {
                    ranges[newRangeCounter] = ForTask2d::Range(ForTask::Range::split(ranges[1]._rows), ForTask::Range::split(ranges[0]._cols));
                    ++newRangeCounter;

                    Task::Status synchStatus;
                    simulation::TaskScheduler* scheduler = simulation::TaskScheduler::getInstance();
                    scheduler->addTask(new InternalForTask2d(_forTask, ranges[0], &synchStatus));
                    scheduler->addTask(new InternalForTask2d(_forTask, ranges[1], &synchStatus));
                    scheduler->workUntilDone(&synchStatus);

                    scheduler->addTask(new InternalForTask2d(_forTask, ForTask2d::Range(ranges[2]), _status));
                    this->run();
                }
                else if (newRangeCounter == 1)
                {
                    simulation::TaskScheduler* scheduler = simulation::TaskScheduler::getInstance();
                    scheduler->addTask(new InternalForTask2d(_forTask, ranges[0], _status));
                    this->run();

                }
                else
                {
                    _forTask->operator()(_range);
                }
            }

        public:
            const ForTask2d* _forTask;
            ForTask2d::Range _range;
            ForTask2d::Partition _partition;
        };

        
        void ParallelFor2d(ForTask2d& task, const ForTask2d::Range& range, ForTask2d::Partition partition)
        {
            simulation::Task::Status status;
            InternalForTask2d* internalTask = new InternalForTask2d(&task, range, &status);
            simulation::TaskScheduler* currentScheduler = TaskScheduler::getInstance();
            currentScheduler->addTask(internalTask);
            currentScheduler->workUntilDone(&status);
        }


	} // namespace simulation

} // namespace sofa
