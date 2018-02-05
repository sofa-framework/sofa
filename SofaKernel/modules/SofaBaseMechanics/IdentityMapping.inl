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
#ifndef SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_INL

#include <SofaBaseMechanics/IdentityMapping.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/VecTypes.h>


namespace sofa
{

namespace component
{

namespace mapping
{


template<class TIn, class TOut>
void IdentityMapping<TIn, TOut>::init()
{
    const unsigned n = this->fromModel->getSize();

    this->toModel->resize( n );

    Inherit::init();


    // build J
    {
        static const unsigned N = std::min<unsigned>(NIn, NOut);

        J.compressedMatrix.resize( n*NOut, n*NIn );
        J.compressedMatrix.reserve( n*N );

        for( size_t i=0 ; i<n ; ++i )
        {
            for(unsigned r = 0; r < N; ++r)
            {
                const unsigned row = NOut * i + r;
                J.compressedMatrix.startVec( row );
                const unsigned col = NIn * i + r;
                J.compressedMatrix.insertBack( row, col ) = (OutReal)1;
            }
        }
        J.compressedMatrix.finalize();
    }

}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/, Data<VecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<VecCoord> > out = dOut;
    helper::ReadAccessor< Data<InVecCoord> > in = dIn;

    for(unsigned int i=0; i<out.size(); i++)
    {
        helper::eq(out[i], in[i]);
    }
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/, Data<VecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    helper::WriteOnlyAccessor< Data<VecDeriv> > out = dOut;
    helper::ReadAccessor< Data<InVecDeriv> > in = dIn;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( !this->maskTo->isActivated() || this->maskTo->getEntry(i) )
            helper::eq(out[i], in[i]);
    }
}

template<class TIn, class TOut>
void IdentityMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/, Data<InVecDeriv>& dOut, const Data<VecDeriv>& dIn)
{
    helper::WriteAccessor< Data<InVecDeriv> > out = dOut;
    helper::ReadAccessor< Data<VecDeriv> > in = dIn;

    for( size_t i=0 ; i<this->maskTo->size() ; ++i)
    {
        if( this->maskTo->getEntry(i) )
            helper::peq(out[i], in[i]);
    }
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv& in = dIn.getValue();

    typename Out::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (typename Out::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        typename Out::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        typename Out::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            typename In::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                InDeriv data;
                helper::eq(data, colIt.val());

                o.addCol(colIt.index(), data);

                ++colIt;
            }
        }
    }

    dOut.endEdit();
}

template <class TIn, class TOut>
void IdentityMapping<TIn, TOut>::handleTopologyChange()
{
    if ( this->toModel && this->fromModel && this->toModel->getSize() != this->fromModel->getSize()) this->init();
}

//template <class TIn, class TOut>
//void IdentityMapping<TIn, TOut>::updateJ()
//{
//    size_t currentHash = this->maskTo->getHash();
//    if(  previousMaskHash!=currentHash )
//    {
//        previousMaskHash = currentHash;

//        assert( this->fromModel->getSize() == this->toModel->getSize());

//        static const unsigned N = std::min<unsigned>(NIn, NOut);

//        J.compressedMatrix.setZero();

//        for( size_t i=0 ; i<this->toModel->getSize() ; ++i )
//        {
//            for(unsigned r = 0; r < N; ++r)
//            {
//                if( this->maskTo->getEntry(i) )
//                {
//                    const unsigned row = NOut * i + r;
//                    const unsigned col = NIn * i + r;
//                    J.compressedMatrix.insert( row, col ) = (OutReal)1;
//                }
//            }
//        }
//    }
//}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* IdentityMapping<TIn, TOut>::getJ()
{
//    updateJ();
    return &J;
}

template <class TIn, class TOut>
const typename IdentityMapping<TIn, TOut>::js_type* IdentityMapping<TIn, TOut>::getJs()
{
//    updateJ();
    return &Js;
}


template<class TIn, class TOut>
void IdentityMapping<TIn, TOut>::updateForceMask()
{
    for( size_t i = 0 ; i<this->maskTo->size() ; ++i )
        if( this->maskTo->getEntry(i) )this->maskFrom->insertEntry( i );

}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
