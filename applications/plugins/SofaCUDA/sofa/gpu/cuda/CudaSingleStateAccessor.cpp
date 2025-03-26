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
#define SOFA_GPU_CUDA_CUDASINGLESTATEACCESSOR_CPP
#include <sofa/gpu/cuda/CudaTypes.h>

#include <sofa/gpu/cuda/CudaSingleStateAccessor.h>

#include <sofa/core/behavior/SingleStateAccessor.inl>


namespace sofa::core::behavior
{

using namespace sofa::gpu::cuda;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec1dTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec2dTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec3d1Types>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec6dTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaRigid2dTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaRigid3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec1fTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec2fTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaVec6fTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaRigid2fTypes>;
template class SOFA_GPU_CUDA_API SingleStateAccessor<CudaRigid3fTypes>;

} // namespace sofa::core::behavior
