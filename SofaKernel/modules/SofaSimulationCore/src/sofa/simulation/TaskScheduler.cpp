#include <sofa/simulation/TaskScheduler.h>

#include <sofa/simulation/DefaultTaskScheduler.h>

//#include <sofa/helper/system/thread/CTime.h>



namespace sofa
{

	namespace simulation
	{
        
        // the order of initialization of these static vars is important
        // the TaskScheduler::_schedulers must be initialized before any call to TaskScheduler::registerScheduler
        std::map<std::string, std::function<TaskScheduler*()> > TaskScheduler::_schedulers;
        std::string TaskScheduler::_currentSchedulerName;
        std::unique_ptr<TaskScheduler> TaskScheduler::_currentScheduler = nullptr;
        
        // register default task scheduler
        const bool DefaultTaskScheduler::isRegistered = TaskScheduler::registerScheduler(DefaultTaskScheduler::name(), &DefaultTaskScheduler::create);
        
        
        TaskScheduler* TaskScheduler::create(const char* name)
        {
            // is already the current scheduler
            std::string nameStr(name);
            if (!nameStr.empty() && _currentSchedulerName == name)
                return _currentScheduler.get();
            
            auto iter = _schedulers.find(name);
            if (iter == _schedulers.end())
            {
                // error scheduler not registered
                // create the default task scheduler
                iter = _schedulers.end();
                --iter;
            }
            
            if (_currentScheduler != nullptr)
            {
                _currentScheduler.reset();
            }
            
            TaskSchedulerCreatorFunction& creatorFunc = iter->second;
            _currentScheduler = std::unique_ptr<TaskScheduler>(creatorFunc());
            
            _currentSchedulerName = iter->first;
            
            Task::setAllocator(_currentScheduler->getTaskAllocator());
            
            return _currentScheduler.get();
        }
        
        
        bool TaskScheduler::registerScheduler(const char* name, std::function<TaskScheduler* ()> creatorFunc)
        {
            _schedulers[name] = creatorFunc;
            return true;
        }
        
        TaskScheduler* TaskScheduler::getInstance()
        {
            if (_currentScheduler == nullptr)
            {
                TaskScheduler::create();// TaskSchedulerDefault::getName());
                _currentScheduler->init();
            }
            
            return _currentScheduler.get();
        }
        
        
        TaskScheduler::~TaskScheduler()
        {
        }

	} // namespace simulation

} // namespace sofa
