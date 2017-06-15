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
#ifndef SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_TASKS_INL
#define SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_TASKS_INL

#include "BeamLinearMapping_mt.h"



namespace sofa
{

namespace component
{

namespace mapping
{
	using namespace sofa::defaulttype;


		template <class TIn, class TOut>
	BeamLinearMapping_mt< TIn, TOut>::applyTask::applyTask( const simulation::Task::Status* status ) 
		: Task( status )
		, _mapping(0)	
		, _in(0)
		, _out(0)
		, _firstPoint(0)
		, _lastPoint(0)

	{
	}


	template <class TIn, class TOut>
	bool BeamLinearMapping_mt< TIn, TOut>::applyTask::run( simulation::WorkerThread* )
	{
		for (size_t i = _firstPoint; i < _lastPoint; ++i )
		{
			Coord inpos = _mapping->points[i];
			int in0 = helper::rfloor(inpos[0]);
			if (in0<0) 
				in0 = 0; 
			else if (in0 > (int)_in->size()-2) 
				in0 = _in->size()-2;
			inpos[0] -= in0;

            const typename In::Coord _in0 = (*_in)[in0];
            const typename In::Coord _in1 = (*_in)[in0+1];
			Real beamLengh = _mapping->beamLength[in0];
			Coord& rotatedPoint0 = _mapping->rotatedPoints0[i];
			Coord& rotatedPoint1 = _mapping->rotatedPoints1[i];

			rotatedPoint0 = _in0.getOrientation().rotate(inpos) * beamLengh;
			Coord out0 = _in0.getCenter() + rotatedPoint0;
			Coord inpos1 = inpos; inpos1[0] -= 1;
			rotatedPoint1 = _in1.getOrientation().rotate(inpos1) * beamLengh;
			Coord out1 = _in1.getCenter() + rotatedPoint1;
			
			Real fact = (Real)inpos[0];
			fact = 3*(fact*fact)-2*(fact*fact*fact);
			(*_out)[i] = out0 * (1-fact) + out1 * (fact);
		}
		return true;
	}


	template <class TIn, class TOut>
	BeamLinearMapping_mt< TIn, TOut>::applyJTask::applyJTask( const simulation::Task::Status* status ) 
		: Task( status )
		, _mapping(0)	
		, _in(0)
		, _out(0)
		, _firstPoint(0)
		, _lastPoint(0)

	{
	}


	template <class TIn, class TOut>
	bool BeamLinearMapping_mt< TIn, TOut>::applyJTask::run( simulation::WorkerThread* )
	{
		for (size_t i = _firstPoint; i < _lastPoint; ++i )
		{

			// out = J in
			// J = [ I -OM^ ]
			//out[i] =  v - cross(rotatedPoints[i],omega);

            defaulttype::Vec<N, typename In::Real> inpos = _mapping->points[i];
			int in0 = helper::rfloor(inpos[0]);
			if (in0<0) 
				in0 = 0; 
			else if (in0 > (int)_in->size()-2) 
				in0 = _in->size()-2;
			inpos[0] -= in0;

            const typename In::Deriv _in0 = (*_in)[in0];
            const typename In::Deriv _in1 = (*_in)[in0+1];
			Coord& rotatedPoint0 = _mapping->rotatedPoints0[i];
			Coord& rotatedPoint1 = _mapping->rotatedPoints1[i];

			Deriv omega0 = getVOrientation( _in0 );
			Deriv out0 = getVCenter( _in0 ) - cross( rotatedPoint0, omega0);
			Deriv omega1 = getVOrientation( _in1 );
			Deriv out1 = getVCenter( _in1 ) - cross( rotatedPoint1, omega1);
			Real fact = (Real)inpos[0];
			fact = 3*(fact*fact)-2*(fact*fact*fact);
			
			(*_out)[i] = out0 * (1-fact) + out1 * (fact);
		}
		return true;
	}


	template <class TIn, class TOut>
	BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask::applyJTmechTask( const simulation::Task::Status* status ) 
		: Task( status )
		, _mapping(0)	
		, _in(0)
		, _out(0)
		, _firstPoint(0)
		, _lastPoint(0)

	{
	}


	template <class TIn, class TOut>
	bool BeamLinearMapping_mt< TIn, TOut>::applyJTmechTask::run( simulation::WorkerThread* )
	{
		for (size_t i = _firstPoint; i < _lastPoint; ++i )
		{

			// out = Jt in
			// Jt = [ I     ]
			//      [ -OM^t ]
			// -OM^t = OM^

			//Deriv f = in[i];
			//v += f;
			//omega += cross(rotatedPoints[i],f);

			defaulttype::Vec<N, typename In::Real> inpos = _mapping->points[i];
			int in0 = helper::rfloor(inpos[0]);
			if (in0<0) 
				in0 = 0; 
			else if (in0 > (int)_out->size()-2) 
				in0 = _out->size()-2;
			inpos[0] -= in0;

            typename In::Deriv& _out0 = (*_out)[in0];
            typename In::Deriv& _out1 = (*_out)[in0+1];
			const Coord& rotatedPoint0 = _mapping->rotatedPoints0[i];
			const Coord& rotatedPoint1 = _mapping->rotatedPoints1[i];

			Deriv f = (*_in)[i];
			Real fact = (Real)inpos[0];
			fact = 3*(fact*fact)-2*(fact*fact*fact);

			getVCenter(_out0) += f * (1-fact);
			getVOrientation(_out0) += cross( rotatedPoint0, f) * (1-fact);
			getVCenter(_out1) += f * (fact);
			getVOrientation(_out1) += cross( rotatedPoint1, f) * (fact);

		}
		return true;
	}


	//template <class TIn, class TOut>
	//BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask::applyJTconstrTask( const simulation::Task::Status* status ) 
	//	: Task( status )
	//	, _mapping(0)	
	//	, _in(0)
	//	, _out(0)
	//	, _firstPoint(0)
	//	, _lastPoint(0)

	//{
	//}


	//template <class TIn, class TOut>
	//bool BeamLinearMapping_mt< TIn, TOut>::applyJTconstrTask::run( simulation::WorkerThread* )
	//{
	//	bool result = true;

	//	for (size_t i = _firstPoint; i < _lastPoint; ++i )
	//	{

	//	}

	//	return true;
	//}





} // namespace mapping

} // namespace component

} // namespace sofa

#endif
