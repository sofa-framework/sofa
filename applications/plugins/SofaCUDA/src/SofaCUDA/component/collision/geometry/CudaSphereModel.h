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
#include <sofa/component/collision/geometry/SphereModel.h>

namespace sofa::gpu::cuda
{

using CudaSphereCollisionModel = sofa::component::collision::geometry::SphereCollisionModel<CudaVec3Types>;
using CudaSphereCollisionModelf1 = sofa::component::collision::geometry::SphereCollisionModel<CudaVec3f1Types>;

using CudaSphere = sofa::component::collision::geometry::TSphere<CudaVec3fTypes>;

} // namespace sofa::gpu::cuda


namespace sofa::component::collision::geometry
{

#if !defined(SOFA_GPU_CUDA_CUDASPHEREMODEL_CPP)
extern template class SOFA_GPU_CUDA_API SphereCollisionModel<sofa::gpu::cuda::CudaVec3fTypes>;
extern template class SOFA_GPU_CUDA_API SphereCollisionModel<sofa::gpu::cuda::CudaVec3f1Types>;
#ifdef SOFA_GPU_CUDA_DOUBLE
extern template class SOFA_GPU_CUDA_API SphereCollisionModel<sofa::gpu::cuda::CudaVec3dTypes>;
extern template class SOFA_GPU_CUDA_API SphereCollisionModel<sofa::gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE
#endif

} // namespace sofa::component::collision
