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
#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/engine/select/BoxROI.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::engine::select
{
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaVec2fTypes>;
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaRigid3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaVec2dTypes>;
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API boxroi::BoxROI<gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::component::engine::select

namespace sofa::gpu::cuda
{
int BoxROICudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::engine::select::boxroi::BoxROI<CudaVec2fTypes> >()
        .add< component::engine::select::boxroi::BoxROI<CudaVec3fTypes> >()
        .add< component::engine::select::boxroi::BoxROI<CudaVec3f1Types> >()
        .add< component::engine::select::boxroi::BoxROI<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::engine::select::boxroi::BoxROI<CudaVec2dTypes> >()
        .add< component::engine::select::boxroi::BoxROI<CudaVec3dTypes> >()
        .add< component::engine::select::boxroi::BoxROI<CudaVec3d1Types> >()
        .add< component::engine::select::boxroi::BoxROI<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace sofa::gpu::cuda

