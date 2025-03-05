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
#include <sofa/component/constraint/projective/LinearVelocityProjectiveConstraint.inl>


#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::constraint::projective
{
template class SOFA_GPU_CUDA_API LinearVelocityProjectiveConstraint<gpu::cuda::CudaVec6fTypes>;
template class SOFA_GPU_CUDA_API LinearVelocityProjectiveConstraint<gpu::cuda::CudaRigid3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class SOFA_GPU_CUDA_API LinearVelocityProjectiveConstraint<gpu::cuda::CudaVec6dTypes>;
template class SOFA_GPU_CUDA_API LinearVelocityProjectiveConstraint<gpu::cuda::CudaRigid3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE

}// namespace sofa::component::constraint::projective

namespace sofa::gpu::cuda
{

int LinearVelocityProjectiveConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< sofa::component::constraint::projective::LinearVelocityProjectiveConstraint<CudaVec6fTypes> >()
        .add< sofa::component::constraint::projective::LinearVelocityProjectiveConstraint<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< sofa::component::constraint::projective::LinearVelocityProjectiveConstraint<CudaVec6dTypes> >()
        .add< sofa::component::constraint::projective::LinearVelocityProjectiveConstraint<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace sofa::gpu::cuda
