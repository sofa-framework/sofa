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
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL

#include "CudaIdentityMapping.h"
#include <SofaBaseMechanics/IdentityMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace sofa::core;
using namespace sofa::core::behavior;
using namespace sofa::gpu::cuda;

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    gpu::cuda::MechanicalObjectCudaVec3f_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT(const core::ConstraintParams * /*cparams*/, Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn)
{
    InMatrixDeriv& out = *dOut.beginEdit();
    const MatrixDeriv & in = dIn.getValue();

    gpu::cuda::CudaVec3fTypes::MatrixDeriv::RowConstIterator rowItEnd = in.end();

    for (gpu::cuda::CudaVec3fTypes::MatrixDeriv::RowConstIterator rowIt = in.begin(); rowIt != rowItEnd; ++rowIt)
    {
        gpu::cuda::CudaVec3fTypes::MatrixDeriv::ColConstIterator colIt = rowIt.begin();
        gpu::cuda::CudaVec3fTypes::MatrixDeriv::ColConstIterator colItEnd = rowIt.end();

        // Creates a constraints if the input constraint is not empty.
        if (colIt != colItEnd)
        {
            gpu::cuda::CudaVec3fTypes::MatrixDeriv::RowIterator o = out.writeLine(rowIt.index());

            while (colIt != colItEnd)
            {
                o.addCol(colIt.index(), colIt.val());

                ++colIt;
            }
        }
    }

    dOut.endEdit();
}

//////// CudaVec3f1

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* /*mparams*/, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* /*mparams*/, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* /*mparams*/, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}



} // namespace mapping

} // namespace component

} // namespace sofa

#endif
