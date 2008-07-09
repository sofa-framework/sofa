/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
#ifndef SOFA_GPU_CUDA_CUDAELLIPSOIDFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAELLIPSOIDFORCEFIELD_INL

#include "CudaEllipsoidForceField.h"
#include <sofa/component/forcefield/EllipsoidForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void EllipsoidForceFieldCuda3f_addForce(unsigned int size, GPUEllipsoid* ellipsoid, void* tmp, void* f, const void* x, const void* v);
    void EllipsoidForceFieldCuda3f_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const void* tmp, void* f, const void* dx, double factor); //, const void* dfdx);

    void EllipsoidForceFieldCuda3f1_addForce(unsigned int size, GPUEllipsoid* ellipsoid, void* tmp, void* f, const void* x, const void* v);
    void EllipsoidForceFieldCuda3f1_addDForce(unsigned int size, GPUEllipsoid* ellipsoid, const void* tmp, void* f, const void* dx, double factor); //, const void* dfdx);

    int EllipsoidForceFieldCuda3f_getNTmp();
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void EllipsoidForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const Coord center = this->center.getValue();
    const Coord r = this->vradius.getValue();
    const Real stiffness = this->stiffness.getValue();
    const Real damping = this->damping.getValue();
    data.ellipsoid.center = center;
    for (int i=0; i<3; ++i)
        data.ellipsoid.inv_r2[i] = 1/(r[i]*r[i]);
    data.ellipsoid.stiffness = stiffness;
    data.ellipsoid.damping = damping;
    f.resize(x.size());
    data.tmp.resize((x.size()+BSIZE*2)*EllipsoidForceFieldCuda3f_getNTmp());
    EllipsoidForceFieldCuda3f_addForce(x.size(), &data.ellipsoid, data.tmp.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    EllipsoidForceFieldCuda3f_addDForce(dx.size(), &data.ellipsoid, data.tmp.deviceRead(), df.deviceWrite(), dx.deviceRead(), kFactor);
}


template <>
void EllipsoidForceField<gpu::cuda::CudaVec3f1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    const Coord center = this->center.getValue();
    const Coord r = this->vradius.getValue();
    const Real stiffness = this->stiffness.getValue();
    const Real damping = this->damping.getValue();
    data.ellipsoid.center = center;
    for (int i=0; i<3; ++i)
        data.ellipsoid.inv_r2[i] = 1/(r[i]*r[i]);
    data.ellipsoid.stiffness = stiffness;
    data.ellipsoid.damping = damping;
    f.resize(x.size());
    data.tmp.resize((x.size()+BSIZE*2)*EllipsoidForceFieldCuda3f_getNTmp());
    EllipsoidForceFieldCuda3f1_addForce(x.size(), &data.ellipsoid, data.tmp.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void EllipsoidForceField<gpu::cuda::CudaVec3f1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    EllipsoidForceFieldCuda3f1_addDForce(dx.size(), &data.ellipsoid, data.tmp.deviceRead(), df.deviceWrite(), dx.deviceRead(), kFactor);
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
