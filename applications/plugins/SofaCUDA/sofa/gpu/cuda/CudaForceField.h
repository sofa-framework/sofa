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

#if !defined(SOFACUDA_CUDAFORCEFIELD_CPP)

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/config.h>

namespace SofaCUDA
{

extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec1fTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec2fTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec3f1Types>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec6fTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaRigid2fTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaRigid3fTypes>;

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec1dTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec2dTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec3d1Types>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaVec6dTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaRigid2dTypes>;
extern template class SOFA_GPU_CUDA_API sofa::core::behavior::ForceField<sofa::gpu::cuda::CudaRigid3dTypes>;
#endif

}

#endif
