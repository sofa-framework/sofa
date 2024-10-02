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

#include <sofa/component/mapping/nonlinear/BaseNonLinearMapping.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraintEigenUtils.h>

namespace sofa::component::mapping::nonlinear
{

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
void BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::
init()
{
    core::Mapping<TIn, TOut>::init();

    m_baseMatrices.resize( 1 );
    m_baseMatrices[0] = &this->m_jacobian;
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
void BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::applyJ(
    const core::MechanicalParams* mparams, DataVecDeriv_t<Out>& out,
    const DataVecDeriv_t<In>& in)
{
    SOFA_UNUSED( mparams );
    if( m_jacobian.rowSize() )
    {
        auto dOutWa = sofa::helper::getWriteOnlyAccessor(out);
        auto dInRa = sofa::helper::getReadAccessor(in);
        m_jacobian.mult(dOutWa.wref(),dInRa.ref());
    }
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
void BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::applyJT(
    const core::MechanicalParams* mparams, DataVecDeriv_t<In>& out,
    const DataVecDeriv_t<Out>& in)
{
    SOFA_UNUSED( mparams );
    if( m_jacobian.rowSize() )
    {
        auto dOutRa = sofa::helper::getReadAccessor(in);
        auto dInWa = sofa::helper::getWriteOnlyAccessor(out);
        m_jacobian.addMultTranspose(dInWa.wref(),dOutRa.ref());
    }
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
void BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::applyJT(
    const core::ConstraintParams* cparams, DataMatrixDeriv_t<In>& out,
    const DataMatrixDeriv_t<Out>& in)
{
    SOFA_UNUSED(cparams);
    auto childMatRa  = sofa::helper::getReadAccessor(in);
    auto parentMatWa = sofa::helper::getWriteAccessor(out);
    addMultTransposeEigen(parentMatWa.wref(), m_jacobian.compressedMatrix, childMatRa.ref());
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
void BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::applyDJT(
    const core::MechanicalParams* mparams, core::MultiVecDerivId parentForceId,
    core::ConstMultiVecDerivId childForceId)
{
    SOFA_UNUSED(childForceId);

    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness )
    {
        return;
    }

    Data<VecDeriv_t<In> >& parentForce = *parentForceId[this->fromModel.get()].write();
    const Data<VecDeriv_t<In> >& parentDisplacement = *mparams->readDx(this->fromModel.get());

    const SReal kFactor = mparams->kFactor();

    if( m_geometricStiffnessMatrix.compressedMatrix.nonZeros() )
    {
        helper::WriteAccessor parentForceAccessor(parentForce);
        helper::ReadAccessor parentDisplacementAccessor(parentDisplacement);
        m_geometricStiffnessMatrix.addMult( parentForceAccessor.wref(), parentDisplacementAccessor.ref(), static_cast<Real>(kFactor) );
    }
    else
    {
        const Data<VecDeriv_t<Out> >& childForce = *mparams->readF(this->toModel.get());

        matrixFreeApplyDJT(mparams, static_cast<Real>(kFactor),
            parentForce, parentDisplacement, childForce);
    }
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
const linearalgebra::BaseMatrix* BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::getK()
{
    return &m_geometricStiffnessMatrix;
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
const type::vector<sofa::linearalgebra::BaseMatrix*>* BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::getJs()
{
    return &m_baseMatrices;
}

template <class TIn, class TOut, bool HasStabilizedGeometricStiffness>
void BaseNonLinearMapping<TIn, TOut, HasStabilizedGeometricStiffness>::
updateK(const core::MechanicalParams* mparams, core::ConstMultiVecDerivId childForceId)
{
    const unsigned geometricStiffness = this->d_geometricStiffness.getValue().getSelectedId();
    if( !geometricStiffness ) { this->m_geometricStiffnessMatrix.resize(0,0); return; }

    const Data<VecDeriv_t<Out> >& childForce = *childForceId[this->toModel.get()].read();

    {
        unsigned int kSize = this->fromModel->getSize();
        m_geometricStiffnessMatrix.resizeBlocks(kSize, kSize);
    }

    doUpdateK(mparams, childForce, m_geometricStiffnessMatrix);

    m_geometricStiffnessMatrix.compress();
}


}
