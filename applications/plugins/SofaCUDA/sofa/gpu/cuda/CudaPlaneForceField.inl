/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL
#define SOFA_GPU_CUDA_CUDAPLANEFORCEFIELD_INL

#include "CudaPlaneForceField.h"
#include <SofaBoundaryCondition/PlaneForceField.inl>

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

#ifdef SOFA_GPU_CUDA_DOUBLE

    void PlaneForceFieldCuda3d_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

    void PlaneForceFieldCuda3d1_addForce(unsigned int size, GPUPlane<double>* plane, void* penetration, void* f, const void* x, const void* v);
    void PlaneForceFieldCuda3d1_addDForce(unsigned int size, GPUPlane<double>* plane, const void* penetration, void* f, const void* dx); //, const void* dfdx);

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
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    m_data.plane.normal = d_planeNormal.getValue();
    m_data.plane.d = d_planeD.getValue();
    m_data.plane.stiffness = d_stiffness.getValue();
    m_data.plane.damping = d_damping.getValue();
    f.resize(x.size());
    m_data.penetration.resize(x.size());
    PlaneForceFieldCuda3f_addForce(x.size(), &m_data.plane, m_data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());
    double stiff = m_data.plane.stiffness;
    m_data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3f_addDForce(dx.size(), &m_data.plane, m_data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    m_data.plane.stiffness = (Real)stiff;

    d_df.endEdit();
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    m_data.plane.normal = d_planeNormal.getValue();
    m_data.plane.d = d_planeD.getValue();
    m_data.plane.stiffness = d_stiffness.getValue();
    m_data.plane.damping = d_damping.getValue();
    f.resize(x.size());
    m_data.penetration.resize(x.size());
    PlaneForceFieldCuda3f1_addForce(x.size(), &m_data.plane, m_data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3f1Types>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());
    double stiff = m_data.plane.stiffness;
    m_data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3f1_addDForce(dx.size(), &m_data.plane, m_data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    m_data.plane.stiffness = (Real)stiff;

    d_df.endEdit();
}

#ifdef SOFA_GPU_CUDA_DOUBLE

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    m_data.plane.normal = d_planeNormal.getValue();
    m_data.plane.d = d_planeD.getValue();
    m_data.plane.stiffness = d_stiffness.getValue();
    m_data.plane.damping = d_damping.getValue();

    f.resize(x.size());
    m_data.penetration.resize(x.size());
    PlaneForceFieldCuda3d_addForce(x.size(), &m_data.plane, m_data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3dTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());
    double stiff = m_data.plane.stiffness;
    m_data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3d_addDForce(dx.size(), &m_data.plane, m_data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    m_data.plane.stiffness = (Real)stiff;

    d_df.endEdit();
}


template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addForce(const core::MechanicalParams* /*mparams*/, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v)
{
    VecDeriv& f = *d_f.beginEdit();
    const VecCoord& x = d_x.getValue();
    const VecDeriv& v = d_v.getValue();

    m_data.plane.normal = d_planeNormal.getValue();
    m_data.plane.d = d_planeD.getValue();
    m_data.plane.stiffness = d_stiffness.getValue();
    m_data.plane.damping = d_damping.getValue();

    f.resize(x.size());
    m_data.penetration.resize(x.size());
    PlaneForceFieldCuda3d1_addForce(x.size(), &m_data.plane, m_data.penetration.deviceWrite(), f.deviceWrite(), x.deviceRead(), v.deviceRead());

    d_f.endEdit();
}

template <>
void PlaneForceField<gpu::cuda::CudaVec3d1Types>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx)
{
    VecDeriv& df = *d_df.beginEdit();
    const VecDeriv& dx = d_dx.getValue();
    Real kFactor = (Real)mparams->kFactorIncludingRayleighDamping(this->rayleighStiffness.getValue());

    df.resize(dx.size());
    double stiff = m_data.plane.stiffness;
    m_data.plane.stiffness *= (Real)kFactor;
    PlaneForceFieldCuda3d1_addDForce(dx.size(), &m_data.plane, m_data.penetration.deviceRead(), df.deviceWrite(), dx.deviceRead());
    m_data.plane.stiffness = (Real)stiff;

    d_df.endEdit();
}

#endif // SOFA_GPU_CUDA_DOUBLE

} // namespace forcefield

} // namespace component

} // namespace sofa

#endif
