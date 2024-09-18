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
#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.inl>
#include <sofa/core/BaseLocalMappingMatrix.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/MechanicalParams.h>
#include <iostream>
#include <sofa/simulation/Node.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::apply(const core::MechanicalParams* mparams,
    DataVecCoord_t<Out>& dOut, const DataVecCoord_t<In>& dIn)
{
    helper::WriteOnlyAccessor< DataVecCoord_t<Out> >  out = dOut;
    const helper::ReadAccessor<DataVecCoord_t<In>> in = dIn;

    size_t size = in.size();
    this->getToModel()->resize( size );
    this->jacobian.resizeBlocks( size, size );
    this->jacobian.reserve( size );

    for( unsigned i=0 ; i<size ; ++i )
    {
        const Real& x = in[i][0];
        out[i][0] = x*x;

        this->jacobian.beginRow(i);
        this->jacobian.insertBack( i, i, 2.0*x );
    }

    this->jacobian.compress();
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::matrixFreeApplyDJT(
    const core::MechanicalParams* mparams, Real kFactor,
    Data<VecDeriv_t<In>>& parentForce,
    const Data<VecDeriv_t<In>>& parentDisplacement,
    const Data<VecDeriv_t<Out>>& childForce)
{
    helper::WriteAccessor parentForceAccessor(parentForce);
    helper::ReadAccessor parentDisplacementAccessor(parentDisplacement);
    helper::ReadAccessor childForceAccessor(childForce);

    const size_t size = parentDisplacementAccessor.size();
    kFactor *= 2.0;

    for(unsigned i=0; i<size; i++ )
    {
        parentForceAccessor[i][0] +=
            parentDisplacementAccessor[i][0] * childForceAccessor[i][0] * kFactor;
    }
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::doUpdateK(const core::MechanicalParams* mparams,
    const Data<VecDeriv_t<Out>>& childForce, SparseKMatrixEigen& matrix)
{
    SOFA_UNUSED(mparams);
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();

    const helper::ReadAccessor childForceAccessor(childForce);
    unsigned int size = this->fromModel->getSize();

    for (size_t i = 0; i < size; ++i)
    {
        matrix.beginRow(i);
        matrix.insertBack( i, i, 2*childForceAccessor[i][0] );
    }
}

template <class TIn, class TOut>
void SquareMapping<TIn, TOut>::buildGeometricStiffnessMatrix(
    sofa::core::GeometricStiffnessMatrix* matrices)
{
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();
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
