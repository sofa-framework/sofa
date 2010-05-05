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
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_INL

#include "CudaIdentityMapping.h"
#include <sofa/component/mapping/IdentityMapping.inl>

namespace sofa
{

namespace gpu
{

namespace cuda
{

extern "C"
{
    void MechanicalObjectCudaVec3f_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f_vPEq(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vAssign(unsigned int size, void* res, const void* a);
    void MechanicalObjectCudaVec3f1_vPEq(unsigned int size, void* res, const void* a);
}

} // namespace cuda

} // namespace gpu

namespace component
{

namespace mapping
{

using namespace gpu::cuda;

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    gpu::cuda::MechanicalObjectCudaVec3f_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}


//////// CudaVec3f1

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in )
{
    gpu::cuda::MechanicalObjectCudaVec3f1_vPEq(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in )
{
    out.fastResize(in.size());
    gpu::cuda::MechanicalObjectCudaVec3f1_vAssign(out.size(), out.deviceWrite(), in.deviceRead());
}

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
