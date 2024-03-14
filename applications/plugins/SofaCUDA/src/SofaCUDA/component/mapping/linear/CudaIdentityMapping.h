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

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/mapping/linear/IdentityMapping.h>

namespace sofa::component::mapping::linear
{

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT(const core::ConstraintParams * cparams, Data<InMatrixDeriv>& dOut, const Data<MatrixDeriv>& dIn);


//////// CudaVec3f1

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
inline void IdentityMapping<gpu::cuda::CudaVec3f1Types, gpu::cuda::CudaVec3f1Types>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );





#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_CPP

using namespace sofa::defaulttype;
using namespace sofa::component::mapping::linear;
using namespace sofa::gpu::cuda;

// CudaVec3fTypes
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, CudaVec3f1Types>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, Vec3Types>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< Vec3Types, CudaVec3fTypes>;

// CudaVec3f1Types
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3f1Types, CudaVec3f1Types>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3f1Types, CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3f1Types, Vec3Types>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< Vec3dTypes, CudaVec3f1Types>;


#ifdef SOFA_GPU_CUDA_DOUBLE
// CudaVec3dTypes
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3dTypes, CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3dTypes, CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3dTypes, Vec3Types>;

extern template class SOFA_GPU_CUDA_API  IdentityMapping< CudaVec3fTypes, CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API  IdentityMapping< Vec3Types, CudaVec3dTypes>;

#endif

#endif


} // namespace sofa::component::mapping::linear
