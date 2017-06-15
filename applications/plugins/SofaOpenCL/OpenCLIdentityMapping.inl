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
#ifndef SOFAOPENCL_OPENCLIDENTITYMAPPING_INL
#define SOFAOPENCL_OPENCLIDENTITYMAPPING_INL

#include "OpenCLIdentityMapping.h"

namespace sofa
{

namespace gpu
{

namespace opencl
{

extern "C"
{
    extern void MechanicalObjectOpenCLVec3f_vAssign(unsigned int size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f_vPEq(unsigned int size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f1_vAssign(unsigned int size, _device_pointer res, const _device_pointer a);
    extern void MechanicalObjectOpenCLVec3f1_vPEq(unsigned int size, _device_pointer res, const _device_pointer a);
}


} // namespace opencl

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::opencl;

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3fTypes, gpu::opencl::OpenCLVec3fTypes>::apply( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit(mparams);
    const InVecCoord& in = dIn.getValue(mparams);
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit(mparams);
}

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3fTypes, gpu::opencl::OpenCLVec3fTypes>::applyJ( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit(mparams);
    const InVecDeriv& in = dIn.getValue(mparams);
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit(mparams);
}

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3fTypes, gpu::opencl::OpenCLVec3fTypes>::applyJT( const core::MechanicalParams* mparams /* PARAMS FIRST */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit(mparams);
    const OutVecDeriv& in = dIn.getValue(mparams);
    gpu::opencl::MechanicalObjectOpenCLVec3f_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit(mparams);
}


//////// OpenCLVec3f1

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3f1Types, gpu::opencl::OpenCLVec3f1Types>::apply( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecCoord& dOut, const InDataVecCoord& dIn )
{
    OutVecCoord& out = *dOut.beginEdit(mparams);
    const InVecCoord& in = dIn.getValue(mparams);
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit(mparams);
}

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3f1Types, gpu::opencl::OpenCLVec3f1Types>::applyJ( const core::MechanicalParams* mparams /* PARAMS FIRST */, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn )
{
    OutVecDeriv& out = *dOut.beginEdit(mparams);
    const InVecDeriv& in = dIn.getValue(mparams);
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit(mparams);
}

template <>
void IdentityMapping<gpu::opencl::OpenCLVec3f1Types, gpu::opencl::OpenCLVec3f1Types>::applyJT( const core::MechanicalParams* mparams /* PARAMS FIRST */, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn )
{
    InVecDeriv& out = *dOut.beginEdit(mparams);
    const OutVecDeriv& in = dIn.getValue(mparams);
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
    dOut.endEdit(mparams);
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
