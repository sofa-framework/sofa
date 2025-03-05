/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#pragma once

#include <sofa/gpu/cuda/CudaTypes.h>
#include <sofa/component/mechanicalload/SphereForceField.h>

namespace sofa::gpu::cuda
{

struct GPUSphere
{
    type::Vec3f center;
    float r;
    float stiffness;
    float damping;
};

} // namespace sofa::gpu::cuda

namespace sofa::component::mechanicalload
{

template <>
class SphereForceFieldInternalData<gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::GPUSphere sphere;
    gpu::cuda::CudaVector<type::Vec4f> penetration;
};

template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void SphereForceField<gpu::cuda::CudaVec3fTypes>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
class SphereForceFieldInternalData<gpu::cuda::CudaVec3f1Types>
{
public:
    gpu::cuda::GPUSphere sphere;
    gpu::cuda::CudaVector<type::Vec4f> penetration;
};

template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addForce(const core::MechanicalParams* mparams, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void SphereForceField<gpu::cuda::CudaVec3f1Types>::addDForce(const core::MechanicalParams* mparams, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

} // namespace sofa::component::mechanicalload
