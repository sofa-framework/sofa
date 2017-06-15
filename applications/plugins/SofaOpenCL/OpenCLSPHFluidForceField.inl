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
#ifndef SOFAOPENCL_OPENCLSPHFLUIDFORCEFIELD_INL
#define SOFAOPENCL_OPENCLSPHFLUIDFORCEFIELD_INL

#include "OpenCLSPHFluidForceField.h"
//#include "CPUSPHFluidForceFieldWithOpenCL.h"
#include <SofaSphFluid/SPHFluidForceField.inl>
//#include "OpenCLSpatialGridContainer.inl"

namespace sofa
{

namespace gpu
{

namespace opencl
{




extern void SPHFluidForceFieldOpenCL3f_computeDensity(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer pos4, const _device_pointer x);
extern void SPHFluidForceFieldOpenCL3f_addForce (unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v);
extern void SPHFluidForceFieldOpenCL3f_addDForce(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3f* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v, const _device_pointer dx);



extern void SPHFluidForceFieldOpenCL3d_computeDensity(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3d* params,_device_pointer pos4, const _device_pointer x);
extern void SPHFluidForceFieldOpenCL3d_addForce (unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3d* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v);
extern void SPHFluidForceFieldOpenCL3d_addDForce(unsigned int size, const _device_pointer cells, const _device_pointer cellGhost, GPUSPHFluid3d* params,_device_pointer f, const _device_pointer pos4, const _device_pointer v, const _device_pointer dx);



} // namespace OpenCL

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::opencl;

template<>
void SPHFluidForceFieldInternalData<gpu::opencl::OpenCLVec3fTypes>::Kernels_computeDensity(int gsize, const _device_pointer cells, const _device_pointer cellGhost,_device_pointer pos4, const _device_pointer x)
{
    SPHFluidForceFieldOpenCL3f_computeDensity(gsize, cells, cellGhost, &params, pos4, x);
}

template<>
void SPHFluidForceFieldInternalData<gpu::opencl::OpenCLVec3fTypes>::Kernels_addForce(int gsize, const _device_pointer cells, const _device_pointer cellGhost,_device_pointer f, const _device_pointer pos4, const _device_pointer v)
{

    SPHFluidForceFieldOpenCL3f_addForce (gsize, cells, cellGhost, &params, f, pos4, v);
//CPUSPHFluidForceFieldWithOpenCL::addForce(gsize, cells, cellGhost, (CPUSPHFluidForceField::GPUSPHFluid*)&params, f, pos4, v);

}

template<>
void SPHFluidForceFieldInternalData<gpu::opencl::OpenCLVec3fTypes>::Kernels_addDForce(int gsize, const _device_pointer cells, const _device_pointer cellGhost,_device_pointer f, const _device_pointer pos4, const _device_pointer v, const _device_pointer dx)
{
    SPHFluidForceFieldOpenCL3f_addDForce(gsize, cells, cellGhost, &params, f, pos4, v, dx);
}

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    if (grid == NULL) return;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    grid->updateGrid(x);
    data.fillParams(this, kernelType.getValue());
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    data.pos4.recreate(x.size());
    data.Kernels_computeDensity(
        g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
        data.pos4.deviceWrite(), x.deviceRead());

//WARNING: erreur à l'appel v.deviceRead() quand utilisation 2^n trop grand ou BSIZE = 64
    data.Kernels_addForce(
        g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
        f.deviceWrite(), data.pos4.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3fTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    //?
    return;
    if (grid == NULL) return;

    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    sout << "addDForce(" << mparams->kFactor() << "," << mparams->bFactor() << ")" << sendl;
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecDeriv& v = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();
    data.fillParams(this, mparams->kFactor(), mparams->bFactor());
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    data.Kernels_addDForce(
        g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
        df.deviceWrite(), data.pos4.deviceRead(), v.deviceRead(), dx.deviceRead());

    d_df.endEdit();
}


template<>
void SPHFluidForceFieldInternalData<gpu::opencl::OpenCLVec3dTypes>::Kernels_computeDensity(int gsize, const _device_pointer cells, const _device_pointer cellGhost, _device_pointer pos4, const _device_pointer x)
{
    SPHFluidForceFieldOpenCL3d_computeDensity(gsize, cells, cellGhost, &params, pos4, x);
}

template<>
void SPHFluidForceFieldInternalData<gpu::opencl::OpenCLVec3dTypes>::Kernels_addForce(int gsize, const _device_pointer cells, const _device_pointer cellGhost, _device_pointer f, const _device_pointer pos4, const _device_pointer v)
{
    SPHFluidForceFieldOpenCL3d_addForce (gsize, cells, cellGhost, &params, f, pos4, v);
}

template<>
void SPHFluidForceFieldInternalData<gpu::opencl::OpenCLVec3dTypes>::Kernels_addDForce(int gsize, const _device_pointer cells, const _device_pointer cellGhost, _device_pointer f, const _device_pointer pos4, const _device_pointer v, const _device_pointer dx)
{
    SPHFluidForceFieldOpenCL3d_addDForce(gsize, cells, cellGhost, &params, f, pos4, v, dx);
}

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/ /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    if (grid == NULL) return;

    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    grid->updateGrid(x);
    data.fillParams(this, kernelType.getValue());
    f.resize(x.size());
    Grid::Grid* g = grid->getGrid();
    data.pos4.recreate(x.size());
    data.Kernels_computeDensity(
        g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
        data.pos4.deviceWrite(), x.deviceRead());
    data.Kernels_addForce(
        g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
        f.deviceWrite(), data.pos4.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3dTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    if (grid == NULL) return;
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    //const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const VecDeriv& v = this->mstate->read(core::ConstVecDerivId::velocity())->getValue();
    data.fillParams(this, mparams->kFactor(), mparams->bFactor());
    df.resize(dx.size());
    Grid::Grid* g = grid->getGrid();
    data.Kernels_addDForce(
        g->getNbCells(), g->getCellsVector().deviceRead(), g->getCellGhostVector().deviceRead(),
        df.deviceWrite(), data.pos4.deviceRead(), v.deviceRead(), dx.deviceRead());
    d_df.endEdit();
}

template <>
void SPHFluidForceField<gpu::opencl::OpenCLVec3fTypes>::draw(const sofa::core::visual::VisualParams* vparams)
{
    if(!vparams->displayFlags().getShowForceFields())return;
//if (!getContext()->getShowForceFields()) return;
    //if (grid != NULL)
    //	grid->draw(vparams);
    const VecCoord& x = this->mstate->read(core::ConstVecCoordId::position())->getValue();
    const gpu::opencl::OpenCLVector<defaulttype::Vec4f> pos4 = this->data.pos4;
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
