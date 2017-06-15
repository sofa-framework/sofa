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
#ifndef SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_PARALLEL_H
#define SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_PARALLEL_H


#include <SofaMiscMapping/BeamLinearMapping.h>


#include "TaskSchedulerBoost.h"



namespace sofa
{

namespace component
{

namespace mapping
{


template <class TIn, class TOut>
class BeamLinearMapping_mt : public BeamLinearMapping<TIn, TOut>
{
public:
	SOFA_CLASS(SOFA_TEMPLATE2(BeamLinearMapping_mt,TIn,TOut), SOFA_TEMPLATE2(BeamLinearMapping,TIn,TOut) );

    typedef core::Mapping<TIn, TOut> Inherit;
    typedef TIn In;
    typedef TOut Out;
    typedef Out OutDataTypes;
    typedef typename Out::VecCoord VecCoord;
    typedef typename Out::VecDeriv VecDeriv;
    typedef typename Out::Coord Coord;
    typedef typename Out::Deriv Deriv;

    typedef In InDataTypes;
    typedef typename In::Deriv InDeriv;

    typedef typename Coord::value_type Real;

    typedef BeamLinearMapping<TIn, TOut> BeamLinearMappingInOut;
    enum { N    = BeamLinearMappingInOut::N    };
    enum { NIn  = BeamLinearMappingInOut::NIn  };
    enum { NOut = BeamLinearMappingInOut::NOut };

    typedef defaulttype::Mat<N, N, Real> Mat;
    typedef defaulttype::Vec<N, Real> Vector;
    typedef defaulttype::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::component::linearsolver::CompressedRowSparseMatrix<MBloc> MatrixType;
   

    BeamLinearMapping_mt();

    virtual ~BeamLinearMapping_mt();


	virtual void apply(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data< typename Out::VecCoord >& out, const Data< typename In::VecCoord >& in);

    virtual void applyJ(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data< typename Out::VecDeriv >& out, const Data< typename In::VecDeriv >& in);

    virtual void applyJT(const core::MechanicalParams *mparams /* PARAMS FIRST */, Data< typename In::VecDeriv >& out, const Data< typename Out::VecDeriv >& in);

    //virtual void applyJT(const core::ConstraintParams *cparams /* PARAMS FIRST */, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in);


    virtual void init();            // get the interpolation
    virtual void bwdInit();        // get the points
    



public:

	// granularity
	Data<unsigned int> mGrainSize;



private:


	// all tasks here

	class applyTask : public simulation::Task
	{
	
		typedef typename BeamLinearMapping<TIn,TOut>::Out::VecCoord  VecCoord;
		typedef typename BeamLinearMapping<TIn,TOut>::In::VecCoord  InVecCoord;

	public:

		virtual bool run( simulation::WorkerThread* );

	protected:

		applyTask( const simulation::Task::Status* status );

	private:

		BeamLinearMapping_mt<TIn,TOut>* _mapping;
		
		const helper::ReadAccessor< Data< typename In::VecCoord > >* _in;
		helper::WriteAccessor< Data< typename Out::VecCoord > >* _out;

		size_t _firstPoint;
		size_t _lastPoint;

		friend class BeamLinearMapping_mt<TIn,TOut>;
	};

	
	class applyJTask : public simulation::Task
	{
	
		typedef typename BeamLinearMapping<TIn,TOut>::Out::VecDeriv  VecDeriv;
		typedef typename BeamLinearMapping<TIn,TOut>::In::VecDeriv  InVecDeriv;

	public:
	
		applyJTask( const simulation::Task::Status* status );

		virtual bool run( simulation::WorkerThread* );

	private:

		BeamLinearMapping_mt<TIn,TOut>* _mapping;

		const helper::ReadAccessor< Data< typename In::VecDeriv > >* _in;
		helper::WriteAccessor< Data< typename Out::VecDeriv > >* _out;

		size_t _firstPoint;
		size_t _lastPoint;

		friend class BeamLinearMapping_mt<TIn,TOut>;
	};


	class applyJTmechTask : public simulation::Task
	{
		typedef typename BeamLinearMapping<TIn,TOut>::Out::VecDeriv  VecDeriv;
		typedef typename BeamLinearMapping<TIn,TOut>::In::VecDeriv  InVecDeriv;

	public:
		
		applyJTmechTask( const simulation::Task::Status* status );
	
		virtual bool run( simulation::WorkerThread* );

	private:

		BeamLinearMapping_mt<TIn,TOut>* _mapping;

		const helper::ReadAccessor< Data< typename Out::VecDeriv > >* _in;
		helper::WriteAccessor< Data< typename In::VecDeriv > >* _out;

		size_t _firstPoint;
		size_t _lastPoint;

		friend class BeamLinearMapping_mt<TIn,TOut>;
	};


	//class applyJTconstrTask : public simulation::Task
	//{
	//public:
	//	applyJTconstrTask( const simulation::Task::Status* status );

	//	virtual bool run( simulation::WorkerThread* );

	//private:

	//	BeamLinearMapping_mt<TIn,TOut>* _mapping;

	//	helper::WriteAccessor< Data< typename In::MatrixDeriv > >* _out;
	//	const helper::ReadAccessor< Data< typename Out::MatrixDeriv > >* _in;

	//	size_t _firstPoint;
	//	size_t _lastPoint;

	//	friend class BeamLinearMapping_mt<TIn,TOut>;
	//};


	friend class applyTask;
	friend class applyJTask;
	friend class applyJTmechTask;
	//friend class applyJTconstrTask;


};

} // namespace mapping

} // namespace component

} // namespace sofa

#endif  /* SOFA_COMPONENT_MAPPING_BEAMLINEARMAPPING_PARALLEL_H */
