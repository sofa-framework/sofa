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
#ifndef SOFA_COMPONENT_MAPPING_SquareMapping_INL
#define SOFA_COMPONENT_MAPPING_SquareMapping_INL

#include "SquareMapping.h"
#include <sofa/core/visual/VisualParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>

namespace sofa
{

namespace component
{

namespace mapping
{




template <class TIn, class TOut>
SquareMapping<TIn, TOut>::SquareMapping()
    : Inherit()
    , d_geometricStiffness(initData(&d_geometricStiffness, 1u, "geometricStiffness", "0 -> no GS, 1 -> exact GS (default)"))
{
}

template <class TIn, class TOut>
SquareMapping<TIn, TOut>::~SquareMapping()
{
}


template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::init()
{
    baseMatrices.resize( 1 );
    baseMatrices[0] = &jacobian;

    this->Inherit::init();
}


template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::apply(const core::MechanicalParams * /*mparams*/ , Data<OutVecCoord>& dOut, const Data<InVecCoord>& dIn)
{
    helper::WriteOnlyAccessor< Data<OutVecCoord> >  out = dOut;
    helper::ReadAccessor< Data<InVecCoord> >  in = dIn;

    size_t size = in.size();
    this->getToModel()->resize( size );
    jacobian.resizeBlocks( size, size );
    jacobian.reserve( size );

    for( unsigned i=0 ; i<size ; ++i )
    {
        const Real& x = in[i][0];
        out[i][0] = x*x;

        jacobian.beginRow(i);
        jacobian.insertBack( i, i, 2.0*x );
    }

    jacobian.compress();
}


template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyJ(const core::MechanicalParams * /*mparams*/ , Data<OutVecDeriv>& dOut, const Data<InVecDeriv>& dIn)
{
    if( jacobian.rowSize() )
        jacobian.mult(dOut,dIn);
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() )
        jacobian.addMultTranspose(dIn,dOut);
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue();
    if( !geometricStiffness ) return;

    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get(mparams)].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel));  // parent displacement
    SReal kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel));

    if( K.compressedMatrix.nonZeros() )
    {
        K.addMult( parentForce.wref(), parentDisplacement.ref(), (typename In::Real)kfactor );
    }
    else
    {
        size_t size = parentDisplacement.size();
        kfactor *= 2.0;

        for(unsigned i=0; i<size; i++ )
        {
            parentForce[i][0] += childForce[i][0]*kfactor;
        }
    }
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& )
{
//    serr<<"applyJT(const core::ConstraintParams*, Data<InMatrixDeriv>& , const Data<OutMatrixDeriv>& ) is not implemented"<<sendl;
}


template <class TIn, class TOut>
const sofa::defaulttype::BaseMatrix* SquareMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const helper::vector<sofa::defaulttype::BaseMatrix*>* SquareMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::updateK(const core::MechanicalParams *mparams, core::ConstMultiVecDerivId childForceId )
{
    const unsigned& geometricStiffness = d_geometricStiffness.getValue();
    if( !geometricStiffness ) { K.resize(0,0); return; }

    helper::ReadAccessor<Data<OutVecDeriv> > childForce( *childForceId[this->toModel.get(mparams)].read() );

    unsigned int size = this->fromModel->getSize();
    K.resizeBlocks(size,size);
    K.reserve( size );
    for( size_t i=0 ; i<size ; ++i )
    {
        K.beginRow(i);
        K.insertBack( i, i, 2*childForce[i][0] );
    }
    K.compress();
}

template <class TIn, class TOut>
const defaulttype::BaseMatrix* SquareMapping<TIn, TOut>::getK()
{
    return &K;
}



template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::updateForceMask()
{
    for(size_t i=0, iend=this->maskTo->size(); i<iend; ++i )
    {
        if (this->maskTo->getEntry( i ) )
        {
            this->maskFrom->insertEntry( i );
        }
    }
}




} // namespace mapping

} // namespace component

} // namespace sofa

#endif
