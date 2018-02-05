/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef AnimationLoopTasks_h__
#define AnimationLoopTasks_h__

#include "TaskSchedulerBoost.h"

#include <sofa/helper/system/atomic.h>

namespace sofa
{

	// forawrd declaraion
	namespace core { namespace behavior {
		class BaseAnimationLoop;
	} }

	//namespace helper { namespace system {
	//	template<int> class atomic;
	//} }



namespace simulation
{

	using namespace sofa;


	class StepTask : public Task
	{
	public:
		StepTask(core::behavior::BaseAnimationLoop* aloop, const double t, Task::Status* pStatus);
		
		virtual ~StepTask();

		virtual bool run(WorkerThread* );


	private:

		core::behavior::BaseAnimationLoop* animationloop;
		const double dt;

	};




	class InitPerThreadDataTask : public Task
	{

	public:

		//InitPerThreadDataTask(volatile long* atomicCounter, boost::mutex* mutex, TaskStatus* pStatus );
		InitPerThreadDataTask(helper::system::atomic<int>* atomicCounter, boost::mutex* mutex, Task::Status* pStatus );
		
		virtual ~InitPerThreadDataTask();

		virtual bool run(WorkerThread* );


	private:

		boost::mutex*	 IdFactorygetIDMutex;

		//volatile long* mAtomicCounter;
		helper::system::atomic<int>* mAtomicCounter;

	};


} // namespace simulation

} // namespace sofa

#endif // AnimationLoopTasks_h__
