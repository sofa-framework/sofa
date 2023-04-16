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
#include <SofaCUDA/component/solidmechanics/spring/CudaSpringForceField.inl>
#include <sofa/core/behavior/ForceField.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa::core::behavior
{

template class SOFA_GPU_CUDA_API PairInteractionForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API PairInteractionForceField<sofa::gpu::cuda::CudaVec3f1Types>;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API PairInteractionForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API PairInteractionForceField<sofa::gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::core::behavior


namespace sofa::component::solidmechanics::spring
{

template class SOFA_GPU_CUDA_API SpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API StiffSpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API MeshSpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;

template class SOFA_GPU_CUDA_API SpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API StiffSpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API MeshSpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;


#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API SpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API StiffSpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API MeshSpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;

template class SOFA_GPU_CUDA_API SpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class SOFA_GPU_CUDA_API StiffSpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class SOFA_GPU_CUDA_API MeshSpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::component::solidmechanics::spring

namespace sofa::gpu::cuda
{

//int SpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
//.add< component::interactionforcefield::SpringForceField<CudaVec3fTypes> >()
//.add< component::interactionforcefield::SpringForceField<CudaVec3f1Types> >()
//#ifdef SOFA_GPU_CUDA_DOUBLE
//.add< component::interactionforcefield::SpringForceField<CudaVec3dTypes> >()
//.add< component::interactionforcefield::SpringForceField<CudaVec3d1Types> >()
//#endif // SOFA_GPU_CUDA_DOUBLE
//;

int StiffSpringForceFieldCudaClass = sofa::core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< sofa::component::solidmechanics::spring::StiffSpringForceField<CudaVec3fTypes> >()
        .add< sofa::component::solidmechanics::spring::StiffSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< sofa::component::solidmechanics::spring::StiffSpringForceField<CudaVec3dTypes> >()
        .add< sofa::component::solidmechanics::spring::StiffSpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int MeshSpringForceFieldCudaClass = sofa::core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< sofa::component::solidmechanics::spring::MeshSpringForceField<CudaVec3fTypes> >()
        .add< sofa::component::solidmechanics::spring::MeshSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< sofa::component::solidmechanics::spring::MeshSpringForceField<CudaVec3dTypes> >()
        .add< sofa::component::solidmechanics::spring::MeshSpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int TriangleBendingSpringsCudaClass = sofa::core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< sofa::component::solidmechanics::spring::TriangleBendingSprings<CudaVec3fTypes> >()
        .add< sofa::component::solidmechanics::spring::TriangleBendingSprings<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< sofa::component::solidmechanics::spring::TriangleBendingSprings<CudaVec3dTypes> >()
        .add< sofa::component::solidmechanics::spring::TriangleBendingSprings<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int QuadBendingSpringsCudaClass = sofa::core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< sofa::component::solidmechanics::spring::QuadBendingSprings<CudaVec3fTypes> >()
        .add< sofa::component::solidmechanics::spring::QuadBendingSprings<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< sofa::component::solidmechanics::spring::QuadBendingSprings<CudaVec3dTypes> >()
        .add< sofa::component::solidmechanics::spring::QuadBendingSprings<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace sofa::gpu::cuda
