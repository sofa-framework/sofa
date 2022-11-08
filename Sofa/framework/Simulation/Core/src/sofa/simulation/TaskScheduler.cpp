/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/simulation/TaskScheduler.h>

#include <sofa/simulation/DefaultTaskScheduler.h>

namespace sofa::simulation
{

// the order of initialization of these static vars is important
// the TaskScheduler::_schedulers must be initialized before any call to TaskScheduler::registerScheduler
std::map<std::string, TaskScheduler::TaskSchedulerCreatorFunction> TaskScheduler::_schedulers;
std::string TaskScheduler::_currentSchedulerName;
TaskScheduler* TaskScheduler::_currentScheduler = nullptr;
        
// register default task scheduler
const bool DefaultTaskScheduler::isRegistered = TaskScheduler::registerScheduler(DefaultTaskScheduler::name(), &DefaultTaskScheduler::create);
        
        
TaskScheduler* TaskScheduler::create(const char* name)
{
    // is already the current scheduler
    std::string nameStr(name);
    if (!nameStr.empty() && _currentSchedulerName == name)
        return _currentScheduler;
            
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
        delete _currentScheduler;
    }
            
    TaskSchedulerCreatorFunction& creatorFunc = iter->second;
    _currentScheduler = creatorFunc();
            
    _currentSchedulerName = iter->first;
            
    Task::setAllocator(_currentScheduler->getTaskAllocator());
            
    return _currentScheduler;
}
        
        
bool TaskScheduler::registerScheduler(const char* name, TaskSchedulerCreatorFunction creatorFunc)
{
    _schedulers[name] = creatorFunc;
    return true;
}
        
TaskScheduler* TaskScheduler::getInstance()
{
    if (_currentScheduler == nullptr)
    {
        TaskScheduler::create();
        _currentScheduler->init();
    }
            
    return _currentScheduler;
}

} // namespace sofa::simulation
