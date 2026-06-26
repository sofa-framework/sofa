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
#pragma once

#include <sofa/component/mapping/linear/BeamLinearMapping.h>

#include <sofa/simulation/task/CpuTask.h>


namespace multithreading::component::mapping::linear
{

template <class TIn, class TOut>
class BeamLinearMapping_mt : public sofa::component::mapping::linear::BeamLinearMapping<TIn, TOut>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(BeamLinearMapping_mt,TIn,TOut), SOFA_TEMPLATE2(sofa::component::mapping::linear::BeamLinearMapping,TIn,TOut) );

    typedef sofa::core::Mapping<TIn, TOut> Inherit;
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

    typedef sofa::component::mapping::linear::BeamLinearMapping<TIn, TOut> BeamLinearMappingInOut;
    enum { N    = BeamLinearMappingInOut::N    };
    enum { NIn  = BeamLinearMappingInOut::NIn  };
    enum { NOut = BeamLinearMappingInOut::NOut };

    typedef sofa::type::Mat<N, N, Real> Mat;
    typedef sofa::type::Vec<N, Real> Vector;
    typedef sofa::type::Mat<NOut, NIn, Real> MBloc;
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<MBloc> MatrixType;


    BeamLinearMapping_mt();

    virtual ~BeamLinearMapping_mt();


    virtual void apply(const sofa::core::MechanicalParams *mparams /* PARAMS FIRST */, sofa::Data< typename Out::VecCoord >& out, const sofa::Data< typename In::VecCoord >& in) override;

    virtual void applyJ(const sofa::core::MechanicalParams *mparams /* PARAMS FIRST */, sofa::Data< typename Out::VecDeriv >& out, const sofa::Data< typename In::VecDeriv >& in) override;

    virtual void applyJT(const sofa::core::MechanicalParams *mparams /* PARAMS FIRST */, sofa::Data< typename In::VecDeriv >& out, const sofa::Data< typename Out::VecDeriv >& in) override;

    //virtual void applyJT(const core::ConstraintParams *cparams /* PARAMS FIRST */, Data< typename In::MatrixDeriv >& out, const Data< typename Out::MatrixDeriv >& in);


    virtual void init() override;            // get the interpolation
    virtual void bwdInit() override;        // get the points




public:

    // granularity
    sofa::Data<unsigned int> mGrainSize; ///< minimum number of Beam points for task creation



private:


    // all tasks here

    class applyTask : public sofa::simulation::CpuTask
    {

        typedef typename sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::Out::VecCoord  VecCoord;
        typedef typename sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::In::VecCoord  InVecCoord;

    public:

        MemoryAlloc run() final;

    protected:

        applyTask( sofa::simulation::CpuTask::Status* status );

    private:

        BeamLinearMapping_mt<TIn,TOut>* _mapping;

        const sofa::helper::ReadAccessor< sofa::Data< typename In::VecCoord > >* _in;
        sofa::helper::WriteAccessor< sofa::Data< typename Out::VecCoord > >* _out;

        size_t _firstPoint;
        size_t _lastPoint;

        friend class BeamLinearMapping_mt<TIn,TOut>;
    };


    class applyJTask : public sofa::simulation::CpuTask
    {

        typedef typename sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::Out::VecDeriv  VecDeriv;
        typedef typename sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::In::VecDeriv  InVecDeriv;

    public:

        applyJTask( sofa::simulation::CpuTask::Status* status );

        MemoryAlloc run() final;

    private:

        BeamLinearMapping_mt<TIn,TOut>* _mapping;

        const sofa::helper::ReadAccessor< sofa::Data< typename In::VecDeriv > >* _in;
        sofa::helper::WriteAccessor< sofa::Data< typename Out::VecDeriv > >* _out;

        size_t _firstPoint;
        size_t _lastPoint;

        friend class BeamLinearMapping_mt<TIn,TOut>;
    };


    class applyJTmechTask : public sofa::simulation::CpuTask
    {
        typedef typename sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::Out::VecDeriv  VecDeriv;
        typedef typename sofa::component::mapping::linear::BeamLinearMapping<TIn,TOut>::In::VecDeriv  InVecDeriv;

    public:

        applyJTmechTask( sofa::simulation::CpuTask::Status* status );

        MemoryAlloc run() final;

    private:

        BeamLinearMapping_mt<TIn,TOut>* _mapping;

        const sofa::helper::ReadAccessor< sofa::Data< typename Out::VecDeriv > >* _in;
        sofa::helper::WriteAccessor< sofa::Data< typename In::VecDeriv > >* _out;

        size_t _firstPoint;
        size_t _lastPoint;

        friend class BeamLinearMapping_mt<TIn,TOut>;
    };


    friend class applyTask;
    friend class applyJTask;
    friend class applyJTmechTask;
};

} // namespace sofa::component::mapping
