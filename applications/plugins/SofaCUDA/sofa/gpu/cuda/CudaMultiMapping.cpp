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
#include <sofa/core/MultiMapping.inl>
#include "CudaTypes.h"

namespace sofa::core
{
    using namespace sofa::gpu::cuda;

    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec1fTypes, CudaVec1fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec2fTypes, CudaVec1fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec2fTypes, CudaVec2fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3fTypes, CudaVec3fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3fTypes, CudaVec2fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3fTypes, CudaVec1fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec6fTypes, CudaVec1fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec6fTypes, CudaVec6fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3fTypes, CudaVec1fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3fTypes, CudaVec3fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3fTypes, CudaVec6fTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3fTypes, CudaRigid3fTypes >;

    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3f1Types, CudaVec3f1Types >;
}//namespace sofa::component::mapping
