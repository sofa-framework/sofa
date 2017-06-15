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
#ifndef SOFAOPENCL_OPENCLSPHEREFORCEFIELD_H
#define SOFAOPENCL_OPENCLSPHEREFORCEFIELD_H

#include "OpenCLTypes.h"
#include <SofaBoundaryCondition/SphereForceField.h>

namespace sofa
{

namespace gpu
{

namespace opencl
{


struct GPUSphere
{
    defaulttype::Vec3f center;
    float r;
    float stiffness;
    float damping;
};


} // namespace opencl

} // namespace gpu


namespace component
{

namespace forcefield
{

template <>
class SphereForceFieldInternalData<gpu::opencl::OpenCLVec3fTypes>
{
public:
    gpu::opencl::GPUSphere sphere;
    gpu::opencl::OpenCLVector<defaulttype::Vec4f> penetration;
};

template <>
void SphereForceField<gpu::opencl::OpenCLVec3fTypes>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void SphereForceField<gpu::opencl::OpenCLVec3fTypes>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

template <>
class SphereForceFieldInternalData<gpu::opencl::OpenCLVec3f1Types>
{
public:
    gpu::opencl::GPUSphere sphere;
    gpu::opencl::OpenCLVector<defaulttype::Vec4f> penetration;
};

template <>
void SphereForceField<gpu::opencl::OpenCLVec3f1Types>::addForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_f, const DataVecCoord& d_x, const DataVecDeriv& d_v);

template <>
void SphereForceField<gpu::opencl::OpenCLVec3f1Types>::addDForce(const core::MechanicalParams* mparams /* PARAMS FIRST */, DataVecDeriv& d_df, const DataVecDeriv& d_dx);

} // namespace forcefield

} // namespace component







} // namespace sofa

#endif
