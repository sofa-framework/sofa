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
#ifndef SOFA_GPU_CUDA_CUDAPARTICLESOURCE_INL
#define SOFA_GPU_CUDA_CUDAPARTICLESOURCE_INL

#include "CudaParticleSource.h"
//#include <sofa/component/misc/ParticleSource.inl>
#include <sofa/gpu/cuda/mycuda.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{


#ifdef SOFA_GPU_CUDA_DOUBLE

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace misc
{

using namespace gpu::cuda;


template <>
void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectResponse(VecDeriv& res)
{
    if (!this->mstate) return;
    if (lastparticles.empty()) return;
    //sout << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<sendl;
    double time = getContext()->getTime();
    if (time < f_start.getValue() || time > f_stop.getValue()) return;
    // constraint the last values
    mycudaMemset(((Deriv*)res.deviceWrite())+lastparticles[0], 0, lastparticles.size()*sizeof(Coord));
}

template <>
void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectVelocity(VecDeriv& res)
{
}

template <>
void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectPosition(VecDeriv& res)
{
}


#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectResponse(VecDeriv& res)
{
    if (!this->mstate) return;
    if (lastparticles.empty()) return;
    //sout << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<sendl;
    double time = getContext()->getTime();
    if (time < f_start.getValue() || time > f_stop.getValue()) return;
    // constraint the last values
    mycudaMemset(((Deriv*)res.deviceWrite())+lastparticles[0], 0, lastparticles.size()*sizeof(Coord));
}

template <>
void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectVelocity(VecDeriv& res)
{
}

template <>
void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectPosition(VecDeriv& res)
{
}

#endif // SOFA_GPU_CUDA_DOUBLE


} // namespace misc

} // namespace component

} // namespace sofa

#endif
