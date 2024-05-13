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

#include <SofaCUDA/config.h>
#include <sofa/component/mapping/linear/SubsetMultiMapping.h>

#if !defined(SOFA_COMPONENT_MAPPING_CUDASUBSETMULTIMAPPING_CPP)

#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa::component::mapping::linear
{

using namespace sofa::gpu::cuda;

extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaVec3Types, CudaVec3Types >;
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaVec1Types, CudaVec1Types >;
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaRigid3Types, CudaRigid3Types >;
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaRigid3Types, CudaVec3Types >;

#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaVec3dTypes, CudaVec3dTypes >;
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaVec1dTypes, CudaVec1dTypes >;
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaRigid3dTypes, CudaRigid3dTypes >;
extern template class SOFA_GPU_CUDA_API SubsetMultiMapping< CudaRigid3dTypes, CudaVec3dTypes >;
#endif

} // namespace sofa::component::mapping::linear
#endif



