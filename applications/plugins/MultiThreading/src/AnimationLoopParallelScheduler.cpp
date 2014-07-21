#include "AnimationLoopParallelScheduler.h"

#include "TaskSchedulerBoost.h"
#include "AnimationLoopTasks.h"
#include "DataExchange.h"

#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/PrintVisitor.h>
#include <sofa/simulation/common/FindByTypeVisitor.h>
#include <sofa/simulation/common/ExportGnuplotVisitor.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/AnimateVisitor.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/CollisionBeginEvent.h>
#include <sofa/simulation/common/CollisionEndEvent.h>
#include <sofa/simulation/common/UpdateContextVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/simulation/common/ResetVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <sofa/simulation/common/ExportOBJVisitor.h>
#include <sofa/simulation/common/WriteStateVisitor.h>
#include <sofa/simulation/common/XMLPrintVisitor.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/BehaviorUpdatePositionVisitor.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/UpdateMappingEndEvent.h>
#include <sofa/simulation/common/CleanupVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/simulation/common/UpdateBoundingBoxVisitor.h>
#include <sofa/simulation/common/xml/NodeElement.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/PipeProcess.h>
#include <sofa/helper/AdvancedTimer.h>
#include <sofa/helper/system/atomic.h>

#include <sofa/core/visual/VisualParams.h>

#include <sofa/helper/AdvancedTimer.h>

#include <stdlib.h>
#include <math.h>
#include <algorithm>


#include <boost/pool/pool.hpp>




namespace sofa
{

namespace simulation
{

	SOFA_DECL_CLASS(AnimationLoopParallelScheduler);

	int AnimationLoopParallelSchedulerClass = core::RegisterObject("parallel animation loop, using intel tbb library")
		.add< AnimationLoopParallelScheduler >()
		;






	AnimationLoopParallelScheduler::AnimationLoopParallelScheduler(simulation::Node* _gnode)
		: Inherit()
		, threadNumber(initData(&threadNumber, 0, "threadNumber", "number of thread") )
		, mNbThread(0)
		, gnode(_gnode)
	{
		//assert(gnode);


	}

	AnimationLoopParallelScheduler::~AnimationLoopParallelScheduler()
	{	

	}

	void AnimationLoopParallelScheduler::init()
	{
		if (!gnode)
			gnode = dynamic_cast<simulation::Node*>(this->getContext());

		//TaskScheduler* mScheduler = &TaskScheduler::getInstance(); // boost::shared_ptr<TaskScheduler>(new TaskScheduler());

		if ( threadNumber.getValue() )
		{
			mNbThread = threadNumber.getValue();
		}

		//TaskScheduler::getInstance().stop();

		TaskScheduler::getInstance().start( mNbThread );


		sofa::core::objectmodel::classidT<sofa::core::behavior::ConstraintSolver>();
		sofa::core::objectmodel::classidT<sofa::core::behavior::LinearSolver>();
		sofa::core::objectmodel::classidT<sofa::core::CollisionModel>();

		//simulation::Node* root = dynamic_cast<simulation::Node*>(getContext());
		//if(root == NULL) return;
		//std::vector<core::behavior::ConstraintSolver*> constraintSolver;
		//root->getTreeObjects<core::behavior::ConstraintSolver>(&constraintSolver);


		//TaskScheduler::getInstance().stop();

	}



	void AnimationLoopParallelScheduler::bwdInit()
	{

		initThreadLocalData();

	}


	void AnimationLoopParallelScheduler::reinit()
	{


	}

	void AnimationLoopParallelScheduler::cleanup()
	{
		//TaskScheduler::getInstance().stop();
	}

	void AnimationLoopParallelScheduler::step(const core::ExecParams* params, double dt)
	{

		static boost::pool<> task_pool(sizeof(StepTask));

		if (dt == 0)
			dt = this->gnode->getDt();


		Task::Status status;

		WorkerThread* thread = WorkerThread::getCurrent();	



		typedef Node::Sequence<simulation::Node,true>::iterator ChildIterator;
		for (ChildIterator it = gnode->child.begin(), itend = gnode->child.end(); it != itend; ++it)
		{
			if ( core::behavior::BaseAnimationLoop* aloop = (*it)->getAnimationLoop() )
			{
				thread->addTask( new( task_pool.malloc()) StepTask( aloop, dt, &status ) );

			}

		}

		thread->workUntilDone(&status);



		double startTime = gnode->getTime();
		gnode->setTime ( startTime + dt );

		// exchange data event
        core::DataExchangeEvent ev ( dt );
		PropagateEventVisitor act ( params, &ev );
		gnode->execute ( act );


		// it doesn't call the destructor
		task_pool.purge_memory();
	}


	void AnimationLoopParallelScheduler::initThreadLocalData()
	{

		boost::pool<> task_pool(sizeof(InitPerThreadDataTask));

		//volatile long atomicCounter = TaskScheduler::getInstance().size();// mNbThread;
		helper::system::atomic<int> atomicCounter( TaskScheduler::getInstance().size() );


		boost::mutex  InitPerThreadMutex;

		Task::Status status;


		WorkerThread* pThread = WorkerThread::getCurrent();	
		const int nbThread = TaskScheduler::getInstance().size();

		for (int i=0; i<nbThread; ++i)
		{
			pThread->addTask( new( task_pool.malloc()) InitPerThreadDataTask( &atomicCounter, &InitPerThreadMutex, &status ) );
		}


		pThread->workUntilDone(&status);

		// it doesn't call the destructor
		task_pool.purge_memory();

	}

} // namespace simulation

} // namespace sofa
