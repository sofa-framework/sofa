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
#ifndef SOFA_GPU_CUDA_CUDASPHFLUIDFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDASPHFLUIDFORCEFIELD_INL

#include "CudaSPHFluidForceField.h"
#include <sofa/component/forcefield/SPHFluidForceField.inl>
//#include <sofa/gpu/cuda/CudaSpatialGridContainer.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void SPHFluidForceFieldCuda3f_computeDensity(unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPUSPHFluid3f* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3f_addForce (unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v);
    void SPHFluidForceFieldCuda3f_addDForce(unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v, const void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void SPHFluidForceFieldCuda3d_computeDensity(unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPUSPHFluid3d* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3d_addForce (unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v);
    void SPHFluidForceFieldCuda3d_addDForce(unsigned int size, const void* cellRange, const void* cellGhost, const void* particleIndex, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v, const void* dx);

#endif // SOFA_GPU_CUDA_DOUBLE
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::Kernels_computeDensity(int gsize, const void* cellRange, const void* cellGhost, const void* particleIndex, void* pos4, const void* x)
{
    SPHFluidForceFieldCuda3f_computeDensity(gsize, cellRange, cellGhost, particleIndex, &params, pos4, x);
}

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::Kernels_addForce(int gsize, const void* cellRange, const void* cellGhost, const void* particleIndex, void* f, const void* pos4, const void* v)
{
    SPHFluidForceFieldCuda3f_addForce (gsize, cellRange, cellGhost, particleIndex, &params, f, pos4, v);
}

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::Kernels_addDForce(int gsize, const void* cellRange, const void* cellGhost, const void* particleIndex, void* f, const void* pos4, const void* v, const void* dx)
{
    SPHFluidForceFieldCuda3f_addDForce(gsize, cellRange, cellGhost, particleIndex, &params, f, pos4, v, dx);
}

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (grid == NULL) return;
    grid->updateGrid(x);
    data.fillParams(this);
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    data.pos4.recreate(x.size());
    data.Kernels_computeDensity(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        data.pos4.deviceWrite(), x.deviceRead());
    data.Kernels_addForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        f.deviceWrite(), data.pos4.deviceRead(), v.deviceRead());
}

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double bFactor)
{
    if (grid == NULL) return;
    sout << "addDForce(" << kFactor << "," << bFactor << ")" << sendl;
    //const VecCoord& x = *this->mstate->getX();
    const VecDeriv& v = *this->mstate->getV();
    data.fillParams(this, kFactor, bFactor);
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    data.Kernels_addDForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        df.deviceWrite(), data.pos4.deviceRead(), v.deviceRead(), dx.deviceRead());
}


#ifdef SOFA_GPU_CUDA_DOUBLE


template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3dTypes>::Kernels_computeDensity(int gsize, const void* cellRange, const void* cellGhost, const void* particleIndex, void* pos4, const void* x)
{
    SPHFluidForceFieldCuda3d_computeDensity(gsize, cellRange, cellGhost, particleIndex, &params, pos4, x);
}

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3dTypes>::Kernels_addForce(int gsize, const void* cellRange, const void* cellGhost, const void* particleIndex, void* f, const void* pos4, const void* v)
{
    SPHFluidForceFieldCuda3d_addForce (gsize, cellRange, cellGhost, particleIndex, &params, f, pos4, v);
}

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3dTypes>::Kernels_addDForce(int gsize, const void* cellRange, const void* cellGhost, const void* particleIndex, void* f, const void* pos4, const void* v, const void* dx)
{
    SPHFluidForceFieldCuda3d_addDForce(gsize, cellRange, cellGhost, particleIndex, &params, f, pos4, v, dx);
}

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3dTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    if (grid == NULL) return;
    grid->updateGrid(x);
    data.fillParams(this);
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    data.pos4.recreate(x.size());
    data.Kernels_computeDensity(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        data.pos4.deviceWrite(), x.deviceRead());
    data.Kernels_addForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        f.deviceWrite(), data.pos4.deviceRead(), v.deviceRead());
}

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3dTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double bFactor)
{
    if (grid == NULL) return;
    //const VecCoord& x = *this->mstate->getX();
    const VecDeriv& v = *this->mstate->getV();
    data.fillParams(this, kFactor, bFactor);
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    data.Kernels_addDForce(
        g->getNbCells(), g->getCellRangeVector().deviceRead(), g->getCellGhostVector().deviceRead(), g->getParticleIndexVector().deviceRead(),
        df.deviceWrite(), data.pos4.deviceRead(), v.deviceRead(), dx.deviceRead());
}

#endif // SOFA_GPU_CUDA_DOUBLE



template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::draw()
{
    if (!getContext()->getShowForceFields()) return;
    //if (grid != NULL)
    //	grid->draw();
    const VecCoord& x = *this->mstate->getX();
    const gpu::cuda::CudaVector<defaulttype::Vec4f> pos4 = this->data.pos4;
    if (pos4.empty()) return;
    glDisable(GL_LIGHTING);
    glColor3f(0,1,1);
    glDisable(GL_BLEND);
    glDepthMask(1);
    glPointSize(5);
    glBegin(GL_POINTS);
    for (unsigned int i=0; i<pos4.size(); i++)
    {
        float density = pos4[i][3];
        float f = (float)(density / density0.getValue());
        f = 1+10*(f-1);
        if (f < 1)
        {
            glColor3f(0,1-f,f);
        }
        else
        {
            glColor3f(f-1,0,2-f);
        }
        helper::gl::glVertexT(x[i]);
    }
    glEnd();
    glPointSize(1);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
