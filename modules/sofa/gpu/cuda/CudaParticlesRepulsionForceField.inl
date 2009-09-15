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
#ifndef SOFA_GPU_CUDA_CUDAPARTICLESREPULSIONFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAPARTICLESREPULSIONFORCEFIELD_INL

#include "CudaParticlesRepulsionForceField.h"
#include <sofa/component/forcefield/ParticlesRepulsionForceField.inl>
//#include <sofa/gpu/cuda/CudaSpatialGridContainer.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void ParticlesRepulsionForceFieldCuda3f_addForce (unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPURepulsion3f* repulsion, void* f, const void* x, const void* v );
    void ParticlesRepulsionForceFieldCuda3f_addDForce(unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPURepulsion3f* repulsion, void* f, const void* x, const void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void ParticlesRepulsionForceFieldCuda3d_addForce (unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPURepulsion3d* repulsion, void* f, const void* x, const void* v );
    void ParticlesRepulsionForceFieldCuda3d_addDForce(unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPURepulsion3d* repulsion, void* f, const void* x, const void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (grid == NULL) return;
    grid->updateGrid(x);
    GPURepulsion3f repulsion;
    repulsion.d = distance.getValue();
    repulsion.d2 = repulsion.d*repulsion.d;
    repulsion.stiffness = stiffness.getValue();
    repulsion.damping = damping.getValue();
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    ParticlesRepulsionForceFieldCuda3f_addForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        &repulsion, f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double bFactor)
{
    if (grid == NULL) return;
    const VecCoord& x = *this->mstate->getX();
    df.resize(dx.size());
    GPURepulsion3f repulsion;
    repulsion.d = distance.getValue();
    repulsion.d2 = repulsion.d*repulsion.d;
    repulsion.stiffness = stiffness.getValue()*kFactor;
    repulsion.damping = damping.getValue()*bFactor;
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    ParticlesRepulsionForceFieldCuda3f_addDForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        &repulsion, df.deviceWrite(), x.deviceRead(), dx.deviceRead());
}


#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3dTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (grid == NULL) return;
    grid->updateGrid(x);
    GPURepulsion3d repulsion;
    repulsion.d = distance.getValue();
    repulsion.d2 = repulsion.d*repulsion.d;
    repulsion.stiffness = stiffness.getValue();
    repulsion.damping = damping.getValue();
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    ParticlesRepulsionForceFieldCuda3d_addForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        &repulsion, f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void ParticlesRepulsionForceField<gpu::cuda::CudaVec3dTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double bFactor)
{
    if (grid == NULL) return;
    const VecCoord& x = *this->mstate->getX();
    df.resize(dx.size());
    GPURepulsion3d repulsion;
    repulsion.d = distance.getValue();
    repulsion.d2 = repulsion.d*repulsion.d;
    repulsion.stiffness = stiffness.getValue()*kFactor;
    repulsion.damping = damping.getValue()*bFactor;
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    ParticlesRepulsionForceFieldCuda3d_addDForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        &repulsion, df.deviceWrite(), x.deviceRead(), dx.deviceRead());
}

#endif // SOFA_GPU_CUDA_DOUBLE


} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
