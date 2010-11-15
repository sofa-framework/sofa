/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL

#include "CudaIdentityMapping.h"
#include <sofa/component/mapping/IdentityMapping.inl>

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
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    gpu::cuda::MechanicalObjectCudaVec3f_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

//////// CudaVec3f1

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( OutDataVecCoord& dOut, const InDataVecCoord& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecCoord& out = *dOut.beginEdit();
    const InVecCoord& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( OutDataVecDeriv& dOut, const InDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    OutVecDeriv& out = *dOut.beginEdit();
    const InVecDeriv& in = dIn.getValue();
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}

template <>
void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( InDataVecDeriv& dOut, const OutDataVecDeriv& dIn, const core::MechanicalParams* /*mparams*/ )
{
    InVecDeriv& out = *dOut.beginEdit();
    const OutVecDeriv& in = dIn.getValue();
    gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit();
}


extern template class  IdentityMapping< CudaVec3fTypes, CudaVec3fTypes>;
#ifndef SOFA_DOUBLE
extern template class  IdentityMapping< CudaVec3fTypes, Vec3fTypes>;
extern template class  IdentityMapping< Vec3fTypes, CudaVec3fTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class  IdentityMapping< CudaVec3fTypes, Vec3dTypes>;
extern template class  IdentityMapping< Vec3dTypes, CudaVec3fTypes>;
#endif

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class  IdentityMapping< CudaVec3fTypes, CudaVec3dTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, CudaVec3fTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, CudaVec3dTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, Vec3fTypes>;
extern template class  IdentityMapping< CudaVec3dTypes, Vec3dTypes>;
#ifndef SOFA_DOUBLE
extern template class  IdentityMapping< Vec3dTypes, CudaVec3dTypes>;
#endif
#ifndef SOFA_FLOAT
extern template class  IdentityMapping< Vec3fTypes, CudaVec3dTypes>;
#endif

extern template class  IdentityMapping< CudaVec3d1Types, ExtVec3dTypes >;
extern template class  IdentityMapping< CudaVec3dTypes, ExtVec3dTypes >;
#endif
extern template class  IdentityMapping< CudaVec3f1Types, ExtVec3fTypes >;
extern template class  IdentityMapping< CudaVec3f1Types, CudaVec3f1Types>;
extern template class  IdentityMapping< CudaVec3f1Types, Vec3dTypes>;
extern template class  IdentityMapping< CudaVec3f1Types, Vec3fTypes>;
#ifndef SOFA_FLOAT
extern template class  IdentityMapping< Vec3dTypes, CudaVec3f1Types>;
#endif
#ifndef SOFA_DOUBLE
extern template class  IdentityMapping< Vec3fTypes, ExtVec3fTypes>;
#endif
extern template class  IdentityMapping< CudaVec3f1Types, ExtVec3dTypes >;
extern template class  IdentityMapping< CudaVec3f1Types, CudaVec3fTypes>;
extern template class  IdentityMapping< CudaVec3fTypes, CudaVec3f1Types>;

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
