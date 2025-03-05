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
#include <sofa/gpu/cuda/CudaTypes.h>

namespace sofa::core
{
    using namespace sofa::gpu::cuda;

    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec1Types, CudaVec1Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec2Types, CudaVec1Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3Types, CudaVec3Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3Types, CudaVec2Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3Types, CudaVec1Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec6Types, CudaVec1Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3Types, CudaVec1Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3Types, CudaVec3Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3Types, CudaVec6Types >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3Types, CudaRigid3Types >;

    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3Types, defaulttype::Vec1Types>;
    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3Types, defaulttype::Vec2Types>;
    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3Types, defaulttype::Vec3Types>;

#ifdef SOFA_GPU_CUDA_DOUBLE
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec1dTypes, CudaVec1dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec2dTypes, CudaVec1dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3dTypes, CudaVec3dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3dTypes, CudaVec2dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec3dTypes, CudaVec1dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaVec6dTypes, CudaVec1dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3dTypes, CudaVec1dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3dTypes, CudaVec3dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3dTypes, CudaVec6dTypes >;
    template class SOFA_GPU_CUDA_API MultiMapping< CudaRigid3dTypes, CudaRigid3dTypes >;

    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, defaulttype::Vec1dTypes>;
    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, defaulttype::Vec3dTypes>;
    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, defaulttype::Vec1fTypes>;
    template class SOFA_GPU_CUDA_API MultiMapping<CudaVec3dTypes, defaulttype::Vec3fTypes>;
#endif
} // namespace sofa::core
