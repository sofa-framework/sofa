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
#ifndef SOFA_GPU_CUDA_CUDASUBSETMAPPING_CPP
#define SOFA_GPU_CUDA_CUDASUBSETMAPPING_CPP

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/mapping/linear/CudaSubsetMapping.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mapping::linear
{
using namespace sofa::gpu::cuda;

template class SOFA_GPU_CUDA_API SubsetMapping< CudaVec3fTypes, CudaVec3fTypes >;
template class SOFA_GPU_CUDA_API SubsetMapping< CudaVec3f1Types, CudaVec3f1Types >;
template class SOFA_GPU_CUDA_API SubsetMapping< CudaVec3f1Types, CudaVec3fTypes >;
template class SOFA_GPU_CUDA_API SubsetMapping< CudaVec3fTypes, CudaVec3f1Types >;

} // namespace sofa::component::mapping::linear

namespace sofa::gpu::cuda
{
using namespace sofa::component::mapping::linear;

int SubsetMappingCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< SubsetMapping< CudaVec3fTypes, CudaVec3fTypes > >()
        .add< SubsetMapping< CudaVec3f1Types, CudaVec3f1Types > >()
        .add< SubsetMapping< CudaVec3f1Types, CudaVec3fTypes > >()
        .add< SubsetMapping< CudaVec3fTypes, CudaVec3f1Types > >()
        ;

} // namespace sofa::gpu::cuda

#endif // SOFA_GPU_CUDA_CUDASUBSETMAPPING_CPP
