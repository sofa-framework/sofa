/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_GPU_CUDA_CUDAPARTICLESOURCE_H
#define SOFA_GPU_CUDA_CUDAPARTICLESOURCE_H

#include "CudaTypes.h"
#include <SofaSphFluid/ParticleSource.h>
#include <sofa/core/behavior/ProjectiveConstraintSet.inl>

namespace sofa
{

namespace component
{

namespace misc
{

// template <>
// void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectResponse(VecDeriv& res);

// template <>
// void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectVelocity(VecDeriv& res);

// template <>
// void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectPosition(VecCoord& res);

// #ifdef SOFA_GPU_CUDA_DOUBLE

// template <>
// void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectResponse(VecDeriv& res);

// template <>
// void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectVelocity(VecDeriv& res);

// template <>
// void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectPosition(VecCoord& res);

// #endif // SOFA_GPU_CUDA_DOUBLE

} // namespace misc

} // namespace component

} // namespace sofa

#endif
