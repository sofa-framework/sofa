/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include "CudaLinearMovementConstraint.inl"
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>
namespace sofa
{


namespace core {
namespace behavior {
    template class ProjectiveConstraintSet<gpu::cuda::CudaVec6fTypes>;
    template class ProjectiveConstraintSet<gpu::cuda::CudaRigid3fTypes>;

#ifdef SOFA_GPU_CUDA_DOUBLE
    template class ProjectiveConstraintSet<gpu::cuda::CudaVec6dTypes>;
    template class ProjectiveConstraintSet<gpu::cuda::CudaRigid3dTypes>;
#endif
}

}

// namespace component
// {
//
// namespace projectiveconstraintset
// {
//
// template class LinearMovementConstraint<gpu::cuda::CudaRigid3fTypes>;
//
// }// namespace projectiveconstraintset
//
// }// namespace component

namespace gpu
{

namespace cuda
{


SOFA_DECL_CLASS(CudaLinearMovementConstraint)

int LinearMovementConstraintCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
// .add< component::projectiveconstraintset::LinearMovementConstraint<CudaVec3fTypes> >()
// .add< component::projectiveconstraintset::LinearMovementConstraint<CudaVec3f1Types> >()
        .add< component::projectiveconstraintset::LinearMovementConstraint<CudaVec6fTypes> >()
        .add< component::projectiveconstraintset::LinearMovementConstraint<CudaRigid3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
// .add< component::projectiveconstraintset::LinearMovementConstraint<CudaVec3dTypes> >()
// .add< component::projectiveconstraintset::LinearMovementConstraint<CudaVec3d1Types> >()
        .add< component::projectiveconstraintset::LinearMovementConstraint<CudaVec6dTypes> >()
        .add< component::projectiveconstraintset::LinearMovementConstraint<CudaRigid3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace cuda

} // namespace gpu

} // namespace sofa
