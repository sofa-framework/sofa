#include "TaskScheduler.h"

#include "TaskSchedulerDefault.h"

//#include <sofa/helper/system/thread/CTime.h>



namespace sofa
{

	namespace simulation
	{
        

        std::map<std::string, std::function<TaskScheduler*()> > TaskScheduler::_schedulers;
        std::string TaskScheduler::_currentSchedulerName;
        TaskScheduler* TaskScheduler::_currentScheduler = nullptr;

        TaskScheduler* TaskScheduler::create(const char* name)
        {
            if (_currentSchedulerName == name)
                return _currentScheduler;

            auto iter = _schedulers.find(name);
            if (iter == _schedulers.end())
            {
                // error scheduler not registered
                // create the default task scheduler
            }

            if (_currentScheduler != nullptr)
            {
                delete _currentScheduler;
            }

            TaskSchedulerCreatorFunction& creatorFunc = iter->second;
            _currentScheduler = creatorFunc();
            _currentSchedulerName = iter->first;

            return _currentScheduler;
        }


        void TaskScheduler::registerScheduler(const char* name, std::function<TaskScheduler* ()> creatorFunc)
        {
            _schedulers[name] = creatorFunc;
        }

        TaskScheduler* TaskScheduler::getInstance()
        {
            if (_currentScheduler == nullptr)
            {
                TaskScheduler::registerScheduler(TaskSchedulerDefault::getName(), &TaskSchedulerDefault::create);
                TaskScheduler::create(TaskSchedulerDefault::getName());
            }

            return _currentScheduler;
        }
		


		// called once by each thread used
		// by the TaskScheduler
		bool runThreadSpecificTask(const Task* task )
		{
            std::atomic<int> atomicCounter;
            TaskScheduler* scheduler = TaskScheduler::getInstance();
            atomicCounter = scheduler->getThreadCount();
            
            std::mutex  InitThreadSpecificMutex;
            
            Task::Status status;
            
            const int nbThread = scheduler->getThreadCount();
            
            for (int i=0; i<nbThread; ++i)
            {
                scheduler->addTask( new ThreadSpecificTask( &atomicCounter, &InitThreadSpecificMutex, &status ) );
            }
            
            scheduler->workUntilDone(&status);

			return true;
		}


	} // namespace simulation

} // namespace sofa
