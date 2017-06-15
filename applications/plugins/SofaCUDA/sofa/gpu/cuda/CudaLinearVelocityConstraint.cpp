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
#include <SofaBoundaryCondition/LinearVelocityConstraint.inl>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{

namespace component
{

namespace projectiveconstraintset
{
template class LinearVelocityConstraint<gpu::cuda::CudaVec6fTypes>;
template class LinearVelocityConstraint<gpu::cuda::CudaRigid3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class LinearVelocityConstraint<gpu::cuda::CudaVec6dTypes>;
template class LinearVelocityConstraint<gpu::cuda::CudaRigid3dTypes>;
#endif // SOFA_GPU_CUDA_DOUBLE
}// namespace projectiveconstraintset

}// namespace component

namespace gpu
{

namespace cuda
{

SOFA_DECL_CLASS(CudaLinearVelocityConstraint)

int LinearVelocityConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::projectiveconstraintset::LinearVelocityConstraint<CudaVec6fTypes> >()
        .add< component::projectiveconstraintset::LinearVelocityConstraint<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::projectiveconstraintset::LinearVelocityConstraint<CudaVec6dTypes> >()
        .add< component::projectiveconstraintset::LinearVelocityConstraint<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
