/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#ifndef SOFA_GPU_CUDA_CUDAPARTICLESOURCE_INL
#define SOFA_GPU_CUDA_CUDAPARTICLESOURCE_INL

#include "CudaParticleSource.h"
#include <sofa/gpu/cuda/mycuda.h>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void ParticleSourceCuda3f_fillValues(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, float fx, float fy, float fz);
    void ParticleSourceCuda3f_copyValuesWithOffset(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, const void* src, float fx, float fy, float fz);


#ifdef SOFA_GPU_CUDA_DOUBLE

    void ParticleSourceCuda3d_fillValues(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, double fx, double fy, double fz);
    void ParticleSourceCuda3d_copyValuesWithOffset(unsigned int totalsize, unsigned int subsetsize, void* dest, const void* indices, const void* src, double fx, double fy, double fz);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace misc
{

using namespace gpu::cuda;

// template <>
// void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectResponse(VecDeriv& res)
// {
//     if (!this->mstate) return;
//     const VecIndex& lastparticles = this->lastparticles.getValue();
//     if (lastparticles.empty()) return;
//     //sout << "ParticleSource: projectResponse of last particles ("<<lastparticles<<")."<<sendl;
//     double time = getContext()->getTime();
//     if (time < f_start.getValue() || time > f_stop.getValue()) return;
//     // constraint the last values
//     //mycudaMemset(((Deriv*)res.deviceWrite())+lastparticles[0], 0, lastparticles.size()*sizeof(Deriv));
//     ParticleSourceCuda3f_fillValues(res.size(), lastparticles.size(), res.deviceWrite(), lastparticles.deviceRead(), 0, 0, 0);
// }

// template <>
// void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectVelocity(VecDeriv& res)
// {
//     if (!this->mstate) return;
//     const VecIndex& lastparticles = this->lastparticles.getValue();
//     if (lastparticles.empty()) return;
//     //sout << "ParticleSource: projectVelocity of last particles ("<<lastparticles[0]<<"-"<<lastparticles[lastparticles.size()-1]<<") out of " << res.size() << "."<<sendl;
//     double time = getContext()->getTime();
//     if (time < f_start.getValue() || time > f_stop.getValue()) return;
//     // constraint the last values
//     Deriv vel = f_velocity.getValue();
// #if 1
//     //mycudaMemset(((Deriv*)res.deviceWrite())+lastparticles[0], 0, lastparticles.size()*sizeof(Coord));
//     ParticleSourceCuda3f_fillValues(res.size(), lastparticles.size(), res.deviceWrite(), lastparticles.deviceRead(), vel[0], vel[1], vel[2]);
// #else
//     for (unsigned int s=0; s<lastparticles.size(); s++)
//         if ( lastparticles[s] < res.size() )
//             res[lastparticles[s]] = vel;
// #endif
// }

// template <>
// void ParticleSource<gpu::cuda::CudaVec3fTypes>::projectPosition(VecCoord& res)
// {
//     if (!this->mstate) return;
//     const VecIndex& lastparticles = this->lastparticles.getValue();
//     if (lastparticles.empty()) return;
//     //sout << "ParticleSource: projectVelocity of last particles ("<<lastparticles<<")."<<sendl;
//     double time = getContext()->getTime();
//     if (time < f_start.getValue() || time > f_stop.getValue()) return;
//     // constraint the last values
//     Deriv vel = f_velocity.getValue();
//     vel *= (time-lasttime);
//     //mycudaMemset(((Deriv*)res.deviceWrite())+lastparticles[0], 0, lastparticles.size()*sizeof(Coord));
//     ParticleSourceCuda3f_copyValuesWithOffset(res.size(), lastparticles.size(), res.deviceWrite(), lastparticles.deviceRead(), lastpos.deviceRead(), vel[0], vel[1], vel[2]);
// }


// #ifdef SOFA_GPU_CUDA_DOUBLE

// template <>
// void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectResponse(VecDeriv& res)
// {
//     if (!this->mstate) return;
//     const VecIndex& lastparticles = this->lastparticles.getValue();
//     if (lastparticles.empty()) return;
//     //sout << "ParticleSource: projectResponse of last particle ("<<lastparticle<<")."<<sendl;
//     double time = getContext()->getTime();
//     if (time < f_start.getValue() || time > f_stop.getValue()) return;
//     // constraint the last values
//     mycudaMemset(((Deriv*)res.deviceWrite())+lastparticles[0], 0, lastparticles.size()*sizeof(Coord));
// }

// template <>
// void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectVelocity(VecDeriv& /*res*/)
// {
// }

// template <>
// void ParticleSource<gpu::cuda::CudaVec3dTypes>::projectPosition(VecDeriv& /*res*/)
// {
// }

// #endif // SOFA_GPU_CUDA_DOUBLE

} // namespace misc

} // namespace component

} // namespace sofa

#endif
