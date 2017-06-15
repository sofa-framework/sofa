/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_PARALLEL_INL
#define SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_PARALLEL_INL

#include "BeamLinearMapping_mt.h"

#include "BeamLinearMapping_tasks.inl"


namespace sofa
{

namespace component
{

namespace mapping
{
	using namespace sofa::defaulttype;




	template <class TIn, class TOut>
	BeamLinearMapping_mt< TIn, TOut>::BeamLinearMapping_mt()
		: mGrainSize(initData(&mGrainSize, (unsigned int)32,"granularity", "minimum number of Beam points for task creation" ))
	{
	}


	template <class TIn, class TOut>
	BeamLinearMapping_mt< TIn, TOut>::~BeamLinearMapping_mt()
	{
	}

	
	template <class TIn, class TOut>
	void BeamLinearMapping_mt< TIn, TOut>::init()
	{
		simulation::TaskScheduler::getInstance().start();

		BeamLinearMapping< TIn, TOut>::init();
	}

	template <class TIn, class TOut>
	void BeamLinearMapping_mt< TIn, TOut>::bwdInit()
	{
		BeamLinearMapping< TIn, TOut>::bwdInit();
	}


	template <class TIn, class TOut>
    void BeamLinearMapping_mt< TIn, TOut>::apply(const core::MechanicalParams* mparams /* PARAMS FIRST */, Data<VecCoord>& _out, const Data<typename In::VecCoord>& _in)
	{

		//Inherit::apply(mparams, dOut, dIn);
		boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyTask));

        unsigned int numPoints = this->points.size();

		if ( numPoints >  2*mGrainSize.getValue()  )
		{		
			helper::WriteAccessor< Data< typename Out::VecCoord > > out = _out;
			helper::ReadAccessor< Data< typename In::VecCoord > > in = _in;

			//const InVecCoord& in= _in.getValue();
			//VecCoord& out = *_out.beginEdit();

            this->rotatedPoints0.resize(this->points.size());
            this->rotatedPoints1.resize(this->points.size());
            out.resize(this->points.size());


			// create tasks
			simulation::Task::Status status;
			simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();	

			//const int nbThread = simulation::TaskScheduler::getInstance().size();

			const int taskSize = 2*mGrainSize.getValue();

			int nbTasks = numPoints / taskSize;
			int pointsLeft = numPoints % taskSize;

			for ( int i=0; i<nbTasks; ++i)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyTask( &status );

				task->_mapping = this;
				//task->_mparams = mparams;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = i*taskSize;
				task->_lastPoint = i*taskSize + mGrainSize.getValue();

				thread->addTask( task );

			}
			if ( pointsLeft > 0)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyTask( &status );

				task->_mapping = this;
				//task->_mparams = mparams;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = nbTasks*taskSize;
				task->_lastPoint = nbTasks*taskSize + pointsLeft;

				thread->addTask( task );

			}

			thread->workUntilDone(&status);


			for ( int i=0; i<nbTasks; ++i)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyTask( &status );

				task->_mapping = this;
				//task->_mparams = mparams;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = i*taskSize + mGrainSize.getValue();
				task->_lastPoint = i*taskSize + taskSize;

				thread->addTask( task );

			}

			thread->workUntilDone(&status);

		}
		else
		{

			BeamLinearMapping<TIn,TOut>::apply( mparams, _out, _in );

		}

		// it doesn't call the destructor
		task_pool.purge_memory();

	}



	template <class TIn, class TOut>
	//void AdaptiveBeamMapping< TIn, TOut>::applyJ( typename Out::VecDeriv& out, const typename In::VecDeriv& in )
	void BeamLinearMapping_mt< TIn, TOut>::applyJ(const core::MechanicalParams * params /* PARAMS FIRST */, Data< typename Out::VecDeriv >& _out, const Data< typename In::VecDeriv >& _in)
	{

		boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyJTask));
        unsigned int numPoints = this->points.size();

		if ( numPoints >  2*mGrainSize.getValue()  )
		{		
			helper::WriteAccessor< Data< typename Out::VecDeriv > > out = _out;
			helper::ReadAccessor< Data< typename In::VecDeriv > > in = _in;

			//const InVecDeriv& in= dIn.getValue();
			//VecDeriv& out = *dOut.beginEdit();


            out.resize(this->points.size());

			simulation::Task::Status status;
			simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();	

			const int taskSize = 2*mGrainSize.getValue();

			int nbTasks = numPoints / taskSize;
			int pointsLeft = numPoints % taskSize;

			for ( int i=0; i<nbTasks; ++i)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyJTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyJTask( &status );

				task->_mapping = this;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = i*taskSize;
				task->_lastPoint = i*taskSize + mGrainSize.getValue();

				thread->addTask( task );

			}
			if ( pointsLeft > 0)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyJTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyJTask( &status );

				task->_mapping = this;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = nbTasks*taskSize;
				task->_lastPoint = nbTasks*taskSize + pointsLeft;

				thread->addTask( task );

			}

			thread->workUntilDone(&status);


			for ( int i=0; i<nbTasks; ++i)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyJTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyJTask( &status );

				task->_mapping = this;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = i*taskSize + mGrainSize.getValue();
				task->_lastPoint = i*taskSize + taskSize;

				thread->addTask( task );

			}

			thread->workUntilDone(&status);

		}
		else
		{

			BeamLinearMapping<TIn,TOut>::applyJ( params, _out, _in );

		}

		// it doesn't call the destructor
		task_pool.purge_memory();

	}




	template <class TIn, class TOut>
	void BeamLinearMapping_mt<TIn, TOut>::applyJT(const core::MechanicalParams * mparams /* PARAMS FIRST */, Data< typename In::VecDeriv >& _out, const Data< typename Out::VecDeriv >& _in)
	{

		boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask));

        unsigned int numPoints = this->points.size();

		if ( numPoints >  2*mGrainSize.getValue()  )
		{		
			helper::WriteAccessor< Data< typename In::VecDeriv > > out = _out;
			helper::ReadAccessor< Data< typename Out::VecDeriv > > in = _in;


			simulation::Task::Status status;
			simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();	

			const int taskSize = 2*mGrainSize.getValue();

			int nbTasks = numPoints / taskSize;
			int pointsLeft = numPoints % taskSize;

			for ( int i=0; i<nbTasks; ++i)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask( &status );

				task->_mapping = this;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = i*taskSize;
				task->_lastPoint = i*taskSize + mGrainSize.getValue();

				thread->addTask( task );

			}
			if ( pointsLeft > 0)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask( &status );

				task->_mapping = this;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = nbTasks*taskSize;
				task->_lastPoint = nbTasks*taskSize + pointsLeft;

				thread->addTask( task );

			}

			thread->workUntilDone(&status);


			for ( int i=0; i<nbTasks; ++i)
			{
				typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask* task = 
					new( task_pool.malloc()) typename BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask( &status );

				task->_mapping = this;
				task->_in = &in;
				task->_out = &out;
				task->_firstPoint = i*taskSize + mGrainSize.getValue();
				task->_lastPoint = i*taskSize + taskSize;

				thread->addTask( task );

			}

			thread->workUntilDone(&status);

		}
		else
		{

			BeamLinearMapping<TIn,TOut>::applyJT( mparams, _out, _in );

		}

		// it doesn't call the destructor
		task_pool.purge_memory();

	}


	//// BeamLinearMapping::applyJT( typename In::MatrixDeriv& out, const typename Out::MatrixDeriv& in ) //
	//// this function propagate the constraint through the rigid mapping :
	//// if one constraint along (vector n) with a value (v) is applied on the childModel (like collision model)
	//// then this constraint is transformed by (Jt.n) with value (v) for the rigid model
	//// There is a specificity of this propagateConstraint: we have to find the application point on the childModel
	//// in order to compute the right constaint on the rigidModel.
	//template <class TIn, class TOut>
	//void BeamLinearMapping_mt<TIn, TOut>::applyJT(const core::ConstraintParams * cparams /* PARAMS FIRST */, Data< typename In::MatrixDeriv >& _out, const Data< typename Out::MatrixDeriv >& _in)
	//{
	//	typename In::MatrixDeriv* out = _out.beginEdit();
	//	const typename Out::MatrixDeriv& in = _in.getValue();

	//	const typename In::VecCoord& x = *this->fromModel->getX();

	//	typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

	//	//for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
	//	//{
	//	//	typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
	//	//	typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

	//	//	if (colIt != colItEnd)
	//	//	{
	//	//		typename In::MatrixDeriv::RowIterator o = out->writeLine(rowIt.index());

	//	//		// computation of (Jt.n)
	//	//		for (typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin(); colIt != colItEnd; ++colIt)
	//	//		{
	//	//			unsigned int indexIn = colIt.index();
	//	//			Deriv data = (Deriv) colIt.val();

	//	//			// interpolation
	//	//			Coord inpos = points[indexIn];
	//	//			int in0 = helper::rfloor(inpos[0]);
	//	//			if (in0<0)
	//	//				in0 = 0;
	//	//			else if (in0 > (int)x.size()-2)
	//	//				in0 = x.size()-2;
	//	//			inpos[0] -= in0;
	//	//			Real fact = (Real)inpos[0];
	//	//			fact = (Real)3.0*(fact*fact) - (Real)2.0*(fact*fact*fact);

	//	//			// weighted value of the constraint direction
	//	//			Deriv w_n = data;

	//	//			// Compute the mapped Constraint on the beam nodes
	//	//			InDeriv direction0;
	//	//			getVCenter(direction0) = w_n * (1-fact);
	//	//			getVOrientation(direction0) = cross(rotatedPoints0[indexIn], w_n) * (1-fact);
	//	//			InDeriv direction1;
	//	//			getVCenter(direction1) = w_n * (fact);
	//	//			getVOrientation(direction1) = cross(rotatedPoints1[indexIn], w_n) * (fact);

	//	//			o.addCol(in0, direction0);
	//	//			o.addCol(in0+1, direction1);
	//	//		}
	//	//	}
	//	//}




	//	//static boost::pool<> task_pool(sizeof(BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask));

	//	//unsigned int numPoints = points.size();

	//	//if ( numPoints >  2*mGrainSize.getValue()  )
	//	//{		
	//	//	helper::WriteAccessor< Data< typename In::VecDeriv > > out = _out;
	//	//	helper::ReadAccessor< Data< typename Out::VecDeriv > > in = _in;

	//	//	//const InVecDeriv& in= dIn.getValue();
	//	//	//VecDeriv& out = *dOut.beginEdit();


	//	//	out.resize(points.size());

	//	//	simulation::Task::Status status;
	//	//	simulation::WorkerThread* thread = simulation::WorkerThread::getCurrent();	

	//	//	const int taskSize = 2*mGrainSize.getValue();

	//	//	int nbTasks = numPoints / taskSize;
	//	//	int pointsLeft = numPoints % taskSize;

	//	//	for ( int i=0; i<nbTasks; ++i)
	//	//	{
	//	//		BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask* task = 
	//	//			new( task_pool.malloc()) BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask( &status );

	//	//		task->_mapping = this;
	//	//		task->_in = &in;
	//	//		task->_out = &out;
	//	//		task->_firstPoint = i*taskSize;
	//	//		task->_lastPoint = i*taskSize + mGrainSize.getValue();

	//	//		thread->addTask( task );

	//	//	}
	//	//	if ( pointsLeft > 0)
	//	//	{
	//	//		BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask* task = 
	//	//			new( task_pool.malloc()) BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask( &status );

	//	//		task->_mapping = this;
	//	//		task->_in = &in;
	//	//		task->_out = &out;
	//	//		task->_firstPoint = nbTasks*taskSize;
	//	//		task->_lastPoint = nbTasks*taskSize + pointsLeft;

	//	//		thread->addTask( task );

	//	//	}

	//	//	thread->workUntilDone(&status);


	//	//	for ( int i=0; i<nbTasks; ++i)
	//	//	{
	//	//		BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask* task = 
	//	//			new( task_pool.malloc()) BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask( &status );

	//	//		task->_mapping = this;
	//	//		task->_in = &in;
	//	//		task->_out = &out;
	//	//		task->_firstPoint = i*taskSize + mGrainSize.getValue();
	//	//		task->_lastPoint = i*taskSize + taskSize;

	//	//		thread->addTask( task );

	//	//	}

	//	//	thread->workUntilDone(&status);

	//	//}
	//	//else
	//	{

	//		BeamLinearMapping<TIn,TOut>::applyJT( cparams, _out, _in );

	//	}

	//	// it doesn't call the destructor
	//	//task_pool.purge_memory();


	//}





} // namespace mapping

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_PARALLEL_INL */
