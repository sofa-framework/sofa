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
#include "CudaTypes.h"
#include "CudaParticleSource.inl"
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.h>
#include <SofaSphFluid/ParticleSource.inl>

namespace sofa
{


namespace core::behavior
{
template class ProjectiveConstraintSet<gpu::cuda::CudaVec3fTypes>;
#ifdef SOFA_GPU_CUDA_DOUBLE
template class ProjectiveConstraintSet<gpu::cuda::CudaVec3dTypes>;
#endif
} // namespace core::behavior


namespace gpu::cuda
{



int ParticleSourceCudaClass = core::RegisterObject("Supports GPU-side computations using CUDA")
        .add< component::misc::ParticleSource<CudaVec3fTypes> >()
#ifdef SOFA_GPU_CUDA_DOUBLE
        .add< component::misc::ParticleSource<CudaVec3dTypes> >()
#endif // SOFA_GPU_CUDA_DOUBLE
        ;

} // namespace gpu::cuda


} // namespace sofa
