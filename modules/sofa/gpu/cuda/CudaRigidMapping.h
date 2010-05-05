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
#ifndef SOFA_GPU_CUDA_CUDARIGIDMAPPING_H
#define SOFA_GPU_CUDA_CUDARIGIDMAPPING_H

#include "CudaTypes.h"
#include <sofa/component/mapping/RigidMapping.h>
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
class RigidMappingInternalData<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::CudaVec3fTypes::VecDeriv tmp;
};

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<gpu::cuda::CudaRigid3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaRigid3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::behavior::State<gpu::cuda::CudaRigid3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

//////// Rigid3d ////////

template <>
class RigidMappingInternalData<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::CudaVec3fTypes::VecDeriv tmp;
};

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<defaulttype::Rigid3dTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<defaulttype::Rigid3dTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<defaulttype::Rigid3dTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::behavior::State<defaulttype::Rigid3dTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::behavior::State<defaulttype::Rigid3dTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

//////// Rigid3f ////////

template <>
class RigidMappingInternalData<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::CudaVec3fTypes::VecDeriv tmp;
};

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<defaulttype::Rigid3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<defaulttype::Rigid3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

template <>
void RigidMapping<sofa::core::behavior::MechanicalMapping< sofa::core::behavior::MechanicalState<defaulttype::Rigid3fTypes>, sofa::core::behavior::MechanicalState<gpu::cuda::CudaVec3fTypes> > >::applyJT( In::VecDeriv& out, const Out::VecDeriv& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::behavior::State<defaulttype::Rigid3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::apply( Out::VecCoord& out, const In::VecCoord& in );

template <>
void RigidMapping<sofa::core::Mapping< sofa::core::behavior::State<defaulttype::Rigid3fTypes>, sofa::core::behavior::MappedModel<gpu::cuda::CudaVec3fTypes> > >::applyJ( Out::VecDeriv& out, const In::VecDeriv& in );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
