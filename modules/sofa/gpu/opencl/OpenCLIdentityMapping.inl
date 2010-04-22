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
#ifndef SOFA_GPU_OPENCL_OPENCLIDENTITYMAPPING_INL
#define SOFA_GPU_OPENCL_OPENCLIDENTITYMAPPING_INL

#include "OpenCLIdentityMapping.h"
#include <sofa/component/mapping/IdentityMapping.inl>

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
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3fTypes>, sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    gpu::opencl::MechanicalObjectOpenCLVec3f_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::opencl::OpenCLVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::opencl::OpenCLVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::opencl::OpenCLVec3fTypes>, sofa::core::componentmodel::behavior::MappedModel<gpu::opencl::OpenCLVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}


//////// OpenCLVec3f1

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3f1Types>, sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3f1Types>, sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::componentmodel::behavior::MechanicalMapping< sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3f1Types>, sofa::core::componentmodel::behavior::MechanicalState<gpu::opencl::OpenCLVec3f1Types> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::opencl::OpenCLVec3f1Types>, sofa::core::componentmodel::behavior::MappedModel<gpu::opencl::OpenCLVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::componentmodel::behavior::State<gpu::opencl::OpenCLVec3f1Types>, sofa::core::componentmodel::behavior::MappedModel<gpu::opencl::OpenCLVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::opencl::MechanicalObjectOpenCLVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
