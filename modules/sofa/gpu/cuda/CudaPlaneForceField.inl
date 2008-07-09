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
#ifndef SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL

#include "CudaPlaneForceField.h"
#include <sofa/component/forcefield/PlaneForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{

    void PlaneForceFieldCuda3f_addForce(unsigned int size, GPUPlane<float>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f_addDForce(unsigned int size, GPUPlane<float>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3f1_addForce(unsigned int size, GPUPlane<float>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3f1_addDForce(unsigned int size, GPUPlane<float>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

    void PlaneForceFieldCuda3d_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3d1_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d1_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3f_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3f_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3f1_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3f1_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}

#ifdef SOFA_DEV
#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3d_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3d_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addForce(VecDeriv& f, const VecCoord& x, const VecDeriv& v)
{
    data.plane.normal = planeNormal.getValue();
    data.plane.d = planeD.getValue();
    data.plane.stiffness = stiffness.getValue();
    data.plane.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    PlaneForceFieldCuda3d1_addForce(x.size(), &data.plane, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addDForce(VecDeriv& df, const VecCoord& dx, double kFactor, double /*bFactor*/)
{
    df.resize(dx.size());
    double stiff = data.plane.stiffness;
    data.plane.stiffness *= kFactor;
    PlaneForceFieldCuda3d1_addDForce(dx.size(), &data.plane, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.plane.stiffness = stiff;
}

#endif // SOFA_GPU_CUDA_DOUBLE
#endif // SOFA_DEV

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
