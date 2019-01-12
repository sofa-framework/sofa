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
#ifndef MultiThreadingTask_h__
#define MultiThreadingTask_h__

#include <sofa/config.h>

#include <atomic>
#include <mutex>

#include <boost/pool/singleton_pool.hpp>


namespace sofa
{
	namespace simulation
    {

        class SOFA_SIMULATION_CORE_API Task
        {
        public:

            // Task Status class definition
            class Status
            {
            public:
                Status() : _busy(0) {}

                bool isBusy() const
                {
                    return (_busy.load(std::memory_order_relaxed) > 0);
                }

                int setBusy(bool busy)
                {
                    if (busy)
                    {
                        return _busy.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        return _busy.fetch_sub(1, std::memory_order_relaxed);
                    }
                }

            private:
                std::atomic<int> _busy;
            };


            Task(const Task::Status* status = nullptr);

            virtual ~Task();

        public:

            virtual bool run() = 0;


            // remove from this interface
        public:

            inline Task::Status* getStatus(void) const
            {
                return const_cast<Task::Status*>(_status);
            }

        protected:

            const Task::Status*	_status;

        public:
            int _id;
        };




		// This task is called once by each thread used by the TasScheduler
		// this is useful to initialize the thread specific variables
		class SOFA_SIMULATION_CORE_API ThreadSpecificTask : public Task
		{

		public:

            ThreadSpecificTask(std::atomic<int>* atomicCounter, std::mutex* mutex, const Task::Status* status);

			virtual ~ThreadSpecificTask();

            virtual bool run() final;


        private:

            virtual bool runThreadSpecific() { return true; }

            virtual bool runCriticalThreadSpecific() { return true; }


			std::atomic<int>* _atomicCounter;
			std::mutex*	 _threadSpecificMutex;
		};


	} // namespace simulation

} // namespace sofa



#endif // MultiThreadingTask_h__
