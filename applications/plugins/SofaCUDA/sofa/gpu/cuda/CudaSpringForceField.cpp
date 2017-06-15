/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "CudaTypes.h"
#include "CudaSpringForceField.inl"
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <SofaGeneralObjectInteraction/BoxStiffSpringForceField.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace core
{
namespace behavior {

template class PairInteractionForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class PairInteractionForceField<sofa::gpu::cuda::CudaVec3f1Types>;

#ifdef SOFA_GPU_CUDA_DOUBLE
template class PairInteractionForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class PairInteractionForceField<sofa::gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE

}
}

namespace component
{

namespace interactionforcefield
{

template class SpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3fTypes>;
template class BoxStiffSpringForceField<gpu::cuda::CudaVec3fTypes>;

template class SpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3f1Types>;
template class BoxStiffSpringForceField<gpu::cuda::CudaVec3f1Types>;


#ifdef SOFA_GPU_CUDA_DOUBLE
template class SpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3dTypes>;
template class BoxStiffSpringForceField<gpu::cuda::CudaVec3dTypes>;

template class SpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class StiffSpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class MeshSpringForceField<sofa::gpu::cuda::CudaVec3d1Types>;
template class BoxStiffSpringForceField<gpu::cuda::CudaVec3d1Types>;
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace forcefield

} // namespace component

namespace gpu
{

namespace cuda
{

//SOFA_DECL_CLASS(CudaSpringForceField)
SOFA_DECL_CLASS(CudaBoxStiffSpringForceField)

//int SpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
//.add< component::interactionforcefield::SpringForceField<CudaVec3fTypes> >()
//.add< component::interactionforcefield::SpringForceField<CudaVec3f1Types> >()
//#ifdef SOFA_GPU_CUDA_DOUBLE
//.add< component::interactionforcefield::SpringForceField<CudaVec3dTypes> >()
//.add< component::interactionforcefield::SpringForceField<CudaVec3d1Types> >()
//#endif // SOFA_GPU_CUDA_DOUBLE
//;

int StiffSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::interactionforcefield::StiffSpringForceField<CudaVec3fTypes> >()
        .add< component::interactionforcefield::StiffSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::interactionforcefield::StiffSpringForceField<CudaVec3dTypes> >()
        .add< component::interactionforcefield::StiffSpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int MeshSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::interactionforcefield::MeshSpringForceField<CudaVec3fTypes> >()
        .add< component::interactionforcefield::MeshSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::interactionforcefield::MeshSpringForceField<CudaVec3dTypes> >()
        .add< component::interactionforcefield::MeshSpringForceField<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int TriangleBendingSpringsCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::interactionforcefield::TriangleBendingSprings<CudaVec3fTypes> >()
        .add< component::interactionforcefield::TriangleBendingSprings<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::interactionforcefield::TriangleBendingSprings<CudaVec3dTypes> >()
        .add< component::interactionforcefield::TriangleBendingSprings<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int QuadBendingSpringsCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::interactionforcefield::QuadBendingSprings<CudaVec3fTypes> >()
        .add< component::interactionforcefield::QuadBendingSprings<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::interactionforcefield::QuadBendingSprings<CudaVec3dTypes> >()
        .add< component::interactionforcefield::QuadBendingSprings<CudaVec3d1Types> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int BoxStiffSpringForceFieldCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::interactionforcefield::BoxStiffSpringForceField<CudaVec3fTypes> >()
        .add< component::interactionforcefield::BoxStiffSpringForceField<CudaVec3f1Types> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::interactionforcefield::BoxStiffSpringForceField<CudaVec3dTypes> >()
        .add< component::interactionforcefield::BoxStiffSpringForceField<CudaVec3d1Types> >()
#endif
        ;


} // namespace cuda

} // namespace gpu

} // namespace sofa
