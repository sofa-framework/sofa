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
#define SOFA_GPU_CUDA_CUDAMECHANICALOBJECT_CPP

#include <sofa/gpu/cuda/CudaTypes.h>
#include <SofaCUDA/component/statecontainer/CudaMechanicalObject.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/component/statecontainer/MappedObject.inl>
#include <sofa/core/State.inl>

namespace sofa::core
{
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec1fTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec2fTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec3fTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec3f1Types, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec6fTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaRigid3fTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaRigid2fTypes, V_DERIV, V_READ>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec1dTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec2dTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec3dTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec3d1Types, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaVec6dTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaRigid3dTypes, V_DERIV, V_READ>;
template struct SOFA_GPU_CUDA_API AccumulationVecId<sofa::gpu::cuda::CudaRigid2dTypes, V_DERIV, V_READ>;
#endif // SOFA_GPU_CUDA_DOUBLE
}

namespace sofa::component::statecontainer
{
// template specialization must be in the same namespace as original namespace for GCC 4.1
// g++ 4.1 requires template instantiations to be declared on a parent namespace from the template class.
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec1fTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec2fTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec3fTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec3f1Types>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec6fTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaRigid3fTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaRigid2fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec1dTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec2dTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec3dTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec3d1Types>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaVec6dTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaRigid3dTypes>;
template class SOFA_GPU_CUDA_API MechanicalObject<CudaRigid2dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace sofa::component::statecontainer

namespace sofa::gpu::cuda
{

int MechanicalObjectCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::statecontainer::MechanicalObject<CudaVec1fTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaVec2fTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaVec3fTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaVec3f1Types> >()
        .add< component::statecontainer::MechanicalObject<CudaVec6fTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaRigid3fTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaRigid2fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::statecontainer::MechanicalObject<CudaVec1dTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaVec2dTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaVec3dTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaVec3d1Types> >()
        .add< component::statecontainer::MechanicalObject<CudaVec6dTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaRigid3dTypes> >()
        .add< component::statecontainer::MechanicalObject<CudaRigid2dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

int MappedObjectCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::statecontainer::MappedObject<CudaVec1fTypes> >()
        .add< component::statecontainer::MappedObject<CudaVec2fTypes> >()
        .add< component::statecontainer::MappedObject<CudaVec3fTypes> >()
        .add< component::statecontainer::MappedObject<CudaVec3f1Types> >()
        .add< component::statecontainer::MappedObject<CudaVec6fTypes> >()
        .add< component::statecontainer::MappedObject<CudaRigid3fTypes> >()
        .add< component::statecontainer::MappedObject<CudaRigid2fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::statecontainer::MappedObject<CudaVec1dTypes> >()
        .add< component::statecontainer::MappedObject<CudaVec2dTypes> >()
        .add< component::statecontainer::MappedObject<CudaVec3dTypes> >()
        .add< component::statecontainer::MappedObject<CudaVec3d1Types> >()
        .add< component::statecontainer::MappedObject<CudaVec6dTypes> >()
        .add< component::statecontainer::MappedObject<CudaRigid3dTypes> >()
        .add< component::statecontainer::MappedObject<CudaRigid2dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace sofa::gpu::cuda
