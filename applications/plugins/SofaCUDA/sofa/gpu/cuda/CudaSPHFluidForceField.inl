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
#ifndef SOFA_GPU_CUDA_CUDASPHFLUIDFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDASPHFLUIDFORCEFIELD_INL

#include "CudaSPHFluidForceField.h"
#include <SofaSphFluid/SPHFluidForceField.inl>
#include <sofa/helper/gl/template.h>
//#include <sofa/gpu/cuda/CudaSpatialGridContainer.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void SPHFluidForceFieldCuda3f_computeDensity(int kernelType, int pressureType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3f_addForce (int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v);
//void SPHFluidForceFieldCuda3f_addDForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3f* params, void* f, const void* pos4, const void* v, const void* dx);

#ifdef SOFA_GPU_CUDA_DOUBLE

    void SPHFluidForceFieldCuda3d_computeDensity(int kernelType, int pressureType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* pos4, const void* x);
    void SPHFluidForceFieldCuda3d_addForce (int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v);
//void SPHFluidForceFieldCuda3d_addDForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, unsigned int size, const void* cells, const void* cellGhost, GPUSPHFluid3d* params, void* f, const void* pos4, const void* v, const void* dx);

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
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::Kernels_computeDensity(int kernelType, int pressureType, int gsize, const void* cells, const void* cellGhost, void* pos4, const void* x)
{
    SPHFluidForceFieldCuda3f_computeDensity(kernelType, pressureType, gsize, cells, cellGhost, &params, pos4, x);
}

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::Kernels_addForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, int gsize, const void* cells, const void* cellGhost, void* f, const void* pos4, const void* v)
{
    SPHFluidForceFieldCuda3f_addForce (kernelType, pressureType, viscosityType, surfaceTensionType, gsize, cells, cellGhost, &params, f, pos4, v);
}
/*
template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3fTypes>::Kernels_addDForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, int gsize, const void* cells, const void* cellGhost, void* f, const void* pos4, const void* v, const void* dx)
{
    SPHFluidForceFieldCuda3f_addDForce(kernelType, pressureType, viscosityType, surfaceTensionType, gsize, cells, cellGhost, &params, f, pos4, v, dx);
}
*/
template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    if (grid == NULL) return;

    const int kernelT = kernelType.getValue();
    const int pressureT = pressureType.getValue();
    const Real viscosity = this->viscosity.getValue();
    const int viscosityT = (viscosity == 0) ? 0 : viscosityType.getValue();
    const Real surfaceTension = this->surfaceTension.getValue();
    const int surfaceTensionT = (surfaceTension <= 0) ? 0 : surfaceTensionType.getValue();

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    grid->updateGrid(x);
    data.fillParams(this, kernelT);
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    data.pos4.recreate(x.size());
    data.Kernels_computeDensity( kernelT, pressureT,
            g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
            data.pos4.deviceWrite(), x.deviceRead());
    if (this->f_printLog.getValue())
    {
        sout << "density[" << 0 << "] = " << data.pos4[0][3] << sendl;
        sout << "density[" << data.pos4.size()/2 << "] = " << data.pos4[data.pos4.size()/2][3] << sendl;
    }
    data.Kernels_addForce( kernelT, pressureT, viscosityT, surfaceTensionT,
            g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
            f.deviceWrite(), data.pos4.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /*d_df*/, const DataVecDeriv& /*d_dx*/)
{
    mparams->setKFactorUsed(true);
#if 0
    if (grid == NULL) return;

    const int kernelT = kernelType.getValue();
    const int pressureT = pressureType.getValue();
    const Real viscosity = this->viscosity.getValue();
    const int viscosityT = (viscosity == 0) ? 0 : viscosityType.getValue();
    const Real surfaceTension = this->surfaceTension.getValue();
    const int surfaceTensionT = (surfaceTension <= 0) ? 0 : surfaceTensionType.getValue();

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    //sout << "addDForce(" << mparams->kFactor() << "," << mparams->bFactor() << ")" << sendl;
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecDeriv& v = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();
    data.fillParams(this, kernelT, mparams->kFactor(), mparams->bFactor());
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    data.Kernels_addDForce( kernelT, pressureT, viscosityT, surfaceTensionT,
            g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
            df.deviceWrite(), data.pos4.deviceRead(), v.deviceRead(), dx.deviceRead());

    d_df.endEdit();
#endif
}


#ifdef SOFA_GPU_CUDA_DOUBLE


template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3dTypes>::Kernels_computeDensity(int kernelType, int pressureType, int gsize, const void* cells, const void* cellGhost, void* pos4, const void* x)
{
    SPHFluidForceFieldCuda3d_computeDensity(kernelType, pressureType, gsize, cells, cellGhost, &params, pos4, x);
}

template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3dTypes>::Kernels_addForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, int gsize, const void* cells, const void* cellGhost, void* f, const void* pos4, const void* v)
{
    SPHFluidForceFieldCuda3d_addForce (kernelType, pressureType, viscosityType, surfaceTensionType, gsize, cells, cellGhost, &params, f, pos4, v);
}
/*
template<>
void SPHFluidForceFieldInternalData<gpu::cuda::CudaVec3dTypes>::Kernels_addDForce(int kernelType, int pressureType, int viscosityType, int surfaceTensionType, int gsize, const void* cells, const void* cellGhost, void* f, const void* pos4, const void* v, const void* dx)
{
    SPHFluidForceFieldCuda3d_addDForce(kernelType, pressureType, viscosityType, surfaceTensionType, gsize, cells, cellGhost, &params, f, pos4, v, dx);
}
*/
template <>
void SPHFluidForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    if (grid == NULL) return;

    const int kernelT = kernelType.getValue();
    const int pressureT = pressureType.getValue();
    const Real viscosity = this->viscosity.getValue();
    const int viscosityT = (viscosity == 0) ? 0 : viscosityType.getValue();
    const Real surfaceTension = this->surfaceTension.getValue();
    const int surfaceTensionT = (surfaceTension <= 0) ? 0 : surfaceTensionType.getValue();

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    grid->updateGrid(x);
    data.fillParams(this, kernelT);
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    data.pos4.recreate(x.size());
    data.Kernels_computeDensity( kernelT, pressureT,
            g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
            data.pos4.deviceWrite(), x.deviceRead());
    if (this->f_printLog.getValue())
    {
        sout << "density[" << 0 << "] = " << data.pos4[0][3] << sendl;
        sout << "density[" << data.pos4.size()/2 << "] = " << data.pos4[data.pos4.size()/2][3] << sendl;
    }
    data.Kernels_addForce( kernelT, pressureT, viscosityT, surfaceTensionT,
            g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
            f.deviceWrite(), data.pos4.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void SPHFluidForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& /*d_df*/, const DataVecDeriv& /*d_dx*/)
{
    mparams->setKFactorUsed(true);
#if 0
    if (grid == NULL) return;

    const int kernelT = kernelType.getValue();
    const int pressureT = pressureType.getValue();
    const Real viscosity = this->viscosity.getValue();
    const int viscosityT = (viscosity == 0) ? 0 : viscosityType.getValue();
    const Real surfaceTension = this->surfaceTension.getValue();
    const int surfaceTensionT = (surfaceTension <= 0) ? 0 : surfaceTensionType.getValue();

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecDeriv& v = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();
    data.fillParams(this, mparams->kFactor(), mparams->bFactor());
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    data.Kernels_addDForce( kernelT, pressureT, viscosityT, surfaceTensionT,
            g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
            df.deviceWrite(), data.pos4.deviceRead(), v.deviceRead(), dx.deviceRead());
    d_df.endEdit();
#endif
}

#endif // SOFA_GPU_CUDA_DOUBLE



template <>
void SPHFluidForceField<gpu::cuda::CudaVec3fTypes>::draw(const core::visual::VisualParams* vparams)
{
    if (!vparams->displayFlags().getShowForceFields()) return;
    //if (grid != NULL)
    //	grid->draw(vparams);
    helper::ReadAccessor<VecCoord> x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    helper::ReadAccessor<gpu::cuda::CudaVector<defaulttype::Vec4f> > pos4 = this->data.pos4;
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
