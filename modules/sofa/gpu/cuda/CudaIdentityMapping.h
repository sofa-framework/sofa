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
#ifndef SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H
#define SOFA_GPU_CUDA_CUDAIDENTITYMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/IdentityMapping.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );


//////// CudaVec3f1

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void IdentityMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3f1Types> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3f1Types> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void IdentityMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaVec3f1Types>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3f1Types> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
