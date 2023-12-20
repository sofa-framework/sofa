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
#include <SofaCUDA/component/constraint/projective/CudaLinearMovementProjectiveConstraint.inl>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::core::behavior
{
    template class SOFA_GPU_CUDA_API ProjectiveConstraintSet<gpu::cuda::CudaVec6fTypes>;
    template class SOFA_GPU_CUDA_API ProjectiveConstraintSet<gpu::cuda::CudaRigid3fTypes>;

#ifdef SOFA_GPU_CUDA_DOUBLE
    template class SOFA_GPU_CUDA_API ProjectiveConstraintSet<gpu::cuda::CudaVec6dTypes>;
    template class SOFA_GPU_CUDA_API ProjectiveConstraintSet<gpu::cuda::CudaRigid3dTypes>;
#endif
} // namespace sofa::core::behavior

namespace sofa::gpu::cuda
{


int LinearMovementProjectiveConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
// .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaVec3fTypes> >()
// .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaVec3f1Types> >()
        .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaVec6fTypes> >()
        .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
// .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaVec3dTypes> >()
// .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaVec3d1Types> >()
        .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaVec6dTypes> >()
        .add< component::constraint::projective::LinearMovementProjectiveConstraint<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace sofa::gpu::cuda
