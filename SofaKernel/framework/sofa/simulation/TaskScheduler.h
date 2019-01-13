/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef TaskScheduler_std_h__
#define TaskScheduler_std_h__

#include <sofa/config.h>

#include "Task.h"
#include "Locks.h"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <map>
#include <deque>
#include <string> 


namespace sofa
{

	namespace simulation
	{


        class SOFA_SIMULATION_CORE_API TaskScheduler
        {

        public:

            virtual ~TaskScheduler();

            static TaskScheduler* create(const char* name = "");

            typedef std::function<TaskScheduler* ()> TaskSchedulerCreatorFunction;

            static bool registerScheduler(const char* name, std::function<TaskScheduler* ()> creatorFunc);

            static TaskScheduler* getInstance();

            static const std::string& getCurrentName()  { return _currentSchedulerName; }

            // interface
            virtual void init(const unsigned int nbThread = 0) = 0;

            virtual void stop(void) = 0;

            virtual unsigned int getThreadCount(void) const = 0;

            virtual const char* getCurrentThreadName() = 0;

            // queue task if there is space, and run it otherwise
            virtual bool addTask(Task* task) = 0;

            virtual void workUntilDone(Task::Status* status) = 0;

            virtual void* allocateTask(size_t size) = 0;

            virtual void releaseTask(Task*) = 0;

        protected:

            // factory map: registered schedulers: name, creation function
            static std::map<std::string, std::function<TaskScheduler*()> > _schedulers;

            // current instantiated scheduler
            static std::string _currentSchedulerName;
            static TaskScheduler * _currentScheduler;

            friend class Task;
        };




        SOFA_SIMULATION_CORE_API bool runThreadSpecificTask(const Task *);


	} // namespace simulation

} // namespace sofa


#endif // TaskScheduler_std_h__
