/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_GPU_CUDA_CUDAPARTICLESREPULSIONFORCEFIELD_H
#define SOFA_GPU_CUDA_CUDAPARTICLESREPULSIONFORCEFIELD_H

#include "CudaTypes.h"
#include <sofa/component/forcefield/ParticlesRepulsionForceField.h>
#include <sofa/gpu/cuda/CudaSpatialGridContainer.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

template<class real>
struct GPURepulsion
{
    real d;
    real d2;
    real stiffness;
    real damping;
};

typedef GPURepulsion<float> GPURepulsion3f;
typedef GPURepulsion<double> GPURepulsion3d;

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3fTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3fTypes>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3dTypes>::addForce (VecDeriv& f, const VecCoord& x, const VecDeriv& v);

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3dTypes>::addDForce (VecDeriv& df, const VecDeriv& dx, double kFactor, double bFactor);

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
