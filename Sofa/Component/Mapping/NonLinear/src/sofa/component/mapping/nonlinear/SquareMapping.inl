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

#include <sofa/component/mapping/nonlinear/SquareMapping.h>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
SquareMapping<TIn, TOut>::SquareMapping()
    : Inherit()
    , d_useGeometricStiffnessMatrix(initData(&d_useGeometricStiffnessMatrix, true, "useGeometricStiffnessMatrix",
        "If available (cached), the geometric stiffness matrix is used in order to compute the "
        "product with the parent displacement. Otherwise, the product is computed directly using "
        "the available vectors (matrix-free method)."))
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
    {
        auto dOutWa = sofa::helper::getWriteOnlyAccessor(dOut);
        auto dInRa = sofa::helper::getReadAccessor(dIn);
        jacobian.mult(dOutWa.wref(),dInRa.ref());
    }
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyJT(const core::MechanicalParams * /*mparams*/ , Data<InVecDeriv>& dIn, const Data<OutVecDeriv>& dOut)
{
    if( jacobian.rowSize() )
    {
        auto dOutRa = sofa::helper::getReadAccessor(dOut);
        auto dInWa = sofa::helper::getWriteOnlyAccessor(dIn);
        jacobian.addMultTranspose(dInWa.wref(),dOutRa.ref());
    }
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyDJT(const core::MechanicalParams* mparams, core::MultiVecDerivId parentDfId, core::ConstMultiVecDerivId )
{
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness ) return;

    helper::WriteAccessor<Data<InVecDeriv> > parentForce (*parentDfId[this->fromModel.get()].write());
    helper::ReadAccessor<Data<InVecDeriv> > parentDisplacement (*mparams->readDx(this->fromModel.get()));  // parent displacement
    SReal kfactor = mparams->kFactor();
    helper::ReadAccessor<Data<OutVecDeriv> > childForce (*mparams->readF(this->toModel.get()));

    if(d_useGeometricStiffnessMatrix.getValue() && K.compressedMatrix.nonZeros() )
    {
        K.addMult( parentForce.wref(), parentDisplacement.ref(), (typename In::Real)kfactor );
    }
    else
    {
        const size_t size = parentDisplacement.size();
        kfactor *= 2.0;

        for(unsigned i=0; i<size; i++ )
        {
            parentForce[i][0] += parentDisplacement[i][0] * childForce[i][0]*kfactor;
        }
    }
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::applyJT(const core::ConstraintParams* cparams, Data<InMatrixDeriv>& out, const Data<OutMatrixDeriv>& in)
{
    SOFA_UNUSED(cparams);
    const OutMatrixDeriv& childMat  = sofa::helper::getReadAccessor(in).ref();
    InMatrixDeriv&        parentMat = sofa::helper::getWriteAccessor(out).wref();
    addMultTransposeEigen(parentMat, jacobian.compressedMatrix, childMat);
}


template <class TIn, class TOut>
const sofa::linearalgebra::BaseMatrix* SquareMapping<TIn, TOut>::getJ()
{
    return &jacobian;
}

template <class TIn, class TOut>
const type::vector<sofa::linearalgebra::BaseMatrix*>* SquareMapping<TIn, TOut>::getJs()
{
    return &baseMatrices;
}



template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::updateK(const core::MechanicalParams *mparams, core::ConstMultiVecDerivId childForceId )
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness ) { K.resize(0,0); return; }

    helper::ReadAccessor<Data<OutVecDeriv> > childForce( *childForceId[this->toModel.get()].read() );

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
const linearalgebra::BaseMatrix* SquareMapping<TIn, TOut>::getK()
{
    return &K;
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned geometricStiffness = d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    const auto childForce = this->toModel->readTotalForces();
    const unsigned int size = this->fromModel->getSize();
    const auto dJdx = matrices->getMappingDerivativeIn(this->fromModel).withRespectToPositionsIn(this->fromModel);

    for( sofa::Size i=0 ; i<size ; ++i )
    {
        dJdx(i, i) += 2*childForce[i][0];
    }
}
} // namespace sofa::component::mapping::nonlinear
