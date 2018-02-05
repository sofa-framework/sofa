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
#ifndef SOFA_GPU_CUDA_CUDASPHEREFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDASPHEREFORCEFIELD_INL

#include "CudaSphereForceField.h"
#include <SofaBoundaryCondition/SphereForceField.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void SphereForceFieldCuda3f_addForce(unsigned int size, GPUSphere* sphere, void* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f_addDForce(unsigned int size, GPUSphere* sphere, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void SphereForceFieldCuda3f1_addForce(unsigned int size, GPUSphere* sphere, void* penetration, void* f, const void* x, const void* v);
    void SphereForceFieldCuda3f1_addDForce(unsigned int size, GPUSphere* sphere, const void* penetration, void* f, const void* dx); //, const void* dfdx);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace forcefield
{

using namespace gpu::cuda;


template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    data.sphere.center = sphereCenter.getValue();
    data.sphere.r = sphereRadius.getValue();
    data.sphere.stiffness = stiffness.getValue();
    data.sphere.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    SphereForceFieldCuda3f_addForce(x.size(), &data.sphere, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    df.resize(dx.size());
    double stiff = data.sphere.stiffness;
    data.sphere.stiffness *= mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    SphereForceFieldCuda3f_addDForce(dx.size(), &data.sphere, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.sphere.stiffness = stiff;

    d_df.endEdit();
}


template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    data.sphere.center = sphereCenter.getValue();
    data.sphere.r = sphereRadius.getValue();
    data.sphere.stiffness = stiffness.getValue();
    data.sphere.damping = damping.getValue();
    f.resize(x.size());
    data.penetration.resize(x.size());
    SphereForceFieldCuda3f1_addForce(x.size(), &data.sphere, data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();

    df.resize(dx.size());
    double stiff = data.sphere.stiffness;
    data.sphere.stiffness *= mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());
    SphereForceFieldCuda3f1_addDForce(dx.size(), &data.sphere, data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    data.sphere.stiffness = stiff;

    d_df.endEdit();
}

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
