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
#ifndef SOFA_GPU_CUDA_CUDARIGIDMAPPING_H
#define SOFA_GPU_CUDA_CUDARIGIDMAPPING_H

#include "CudaTypes.h"
#include <SofaRigid/RigidMapping.h>
#include <sofa/core/behavior/MechanicalState.h>

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
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void RigidMapping<gpu::cuda::CudaRigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

//////// Rigid3d ////////
#ifndef SOFA_FLOAT
template <>
class RigidMappingInternalData<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::CudaVec3fTypes::VecDeriv tmp;
};

template <>
void RigidMapping<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void RigidMapping<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void RigidMapping<defaulttype::Rigid3dTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );
#endif

//////// Rigid3f ////////

template <>
class RigidMappingInternalData<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>
{
public:
    gpu::cuda::CudaVec3fTypes::VecDeriv tmp;
};

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::apply( const core::MechanicalParams* mparams, OutDataVecCoord& dOut, const InDataVecCoord& dIn );

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJ( const core::MechanicalParams* mparams, OutDataVecDeriv& dOut, const InDataVecDeriv& dIn );

template <>
void RigidMapping<defaulttype::Rigid3fTypes, gpu::cuda::CudaVec3fTypes>::applyJT( const core::MechanicalParams* mparams, InDataVecDeriv& dOut, const OutDataVecDeriv& dIn );

} // namespace mapping

} // namespace component

} // namespace sofa

#endif
